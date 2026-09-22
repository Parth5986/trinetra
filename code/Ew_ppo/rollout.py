import torch
from torch.distributions import Categorical


def collect_rollout(env, actor, critic, device):
    observations = []
    actions = []
    rewards = []
    dones = []
    log_probs = []
    values = []

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

        action = distribution.sample()

        log_prob = distribution.log_prob(action)

        # Critic

        value = critic(obs_tensor)

        # Environment

        next_obs, reward, terminated, truncated, info = env.step(action.item())

        done = truncated or terminated

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
        values
    )