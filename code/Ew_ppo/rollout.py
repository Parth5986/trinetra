import numpy as np
import torch
from torch.distributions import Categorical


def collect_rollout(env, actor, critic, device, greedy=False):
    observations = []
    actions = []
    rewards = []
    dones = []
    log_probs = []
    values = []
    terminals = []

    obs, info = env.reset()
    done = False

    while not done:

        obs_tensor = torch.tensor(
            obs,
            dtype = torch.float32,
            device = device
        ).reshape(1, -1)

        # Actor

        logits = actor(obs_tensor)

        distribution = Categorical(logits = logits)

        # greedy=True picks the argmax action (deterministic evaluation /
        # deployment). Training always samples (greedy=False) so the policy
        # keeps exploring.
        if greedy:
            action = logits.argmax(dim = -1)
        else:
            action = distribution.sample()

        log_prob = distribution.log_prob(action)

        # Critic

        value = critic(obs_tensor)

        # Environment

        next_obs, reward, terminated, truncated, info = env.step(action.item())

        done = truncated or terminated
        terminals.append(terminated)

        # Store rollout data

        observations.append(obs)
        actions.append(action.detach())
        rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob.detach())
        values.append(value.squeeze().detach())

        obs = next_obs

    return (
        observations,
        actions,
        rewards,
        dones,
        log_probs,
        values,
        terminals,
    )


def collect_rollouts(envs, actor, critic, device, greedy=False):
    """Collect one rollout per environment in lock-step.

    All environments are stepped together with a SINGLE batched actor + critic
    forward per round instead of one forward per timestep per environment, so
    the network cost (the GPU-bound part) is amortized across the batch. The
    env.step calls stay per-environment and sequential.
    """

    n = len(envs)

    observations = [[] for _ in range(n)]
    actions = [[] for _ in range(n)]
    rewards = [[] for _ in range(n)]
    dones = [[] for _ in range(n)]
    log_probs = [[] for _ in range(n)]
    values = [[] for _ in range(n)]
    terminals = [[] for _ in range(n)]

    states = []

    for env in envs:
        obs, info = env.reset()
        states.append(obs)

    active = list(range(n))

    while active:

        obs_batch = torch.tensor(
            np.asarray([states[i] for i in active], dtype = np.float32),
            dtype = torch.float32,
            device = device,
        ).reshape(len(active), -1)

        logits = actor(obs_batch)

        distribution = Categorical(logits = logits)

        if greedy:
            sampled = logits.argmax(dim = -1)
        else:
            sampled = distribution.sample()

        log_prob = distribution.log_prob(sampled)

        value = critic(obs_batch).squeeze()

        next_active = []

        for k, idx in enumerate(active):

            action = sampled[k]

            observations[idx].append(states[idx])
            actions[idx].append(action.detach())
            log_probs[idx].append(log_prob[k].detach())
            values[idx].append(value[k].detach())

            next_obs, reward, terminated, truncated, info = envs[idx].step(
                int(action.item())
            )

            done_now = truncated or terminated

            rewards[idx].append(reward)
            dones[idx].append(done_now)
            terminals[idx].append(terminated)
            states[idx] = next_obs

            if done_now:
                pass
            else:
                next_active.append(idx)

        active = next_active

    rollouts = []

    for i in range(n):
        rollouts.append(
            (
                observations[i],
                actions[i],
                rewards[i],
                dones[i],
                log_probs[i],
                values[i],
                terminals[i],
            )
        )

    return rollouts