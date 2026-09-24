import numpy as np
import torch
from torch.distributions import Categorical


def ppo(
    actor,
    critic,
    observations,
    actions,
    old_log_probs,
    advantages,
    returns,
    epsilon,
    device = "cpu",
    entropy_coef = 0.0,
    old_values = None
):
    # Batch the whole episode's (or multi-env batch's) observations into a
    # single (T, obs_size) tensor so the actor + critic run ONE forward pass
    # per epoch instead of one tiny forward per timestep - the old per-step
    # loop was spending most of its time on GPU/CPU per-call overhead.
    obs_batch = torch.tensor(
        np.asarray(observations, dtype = np.float32).reshape(len(observations), -1),
        dtype = torch.float32,
        device = device,
    )

    logits = actor(obs_batch)              # (T, n_actions)
    values = critic(obs_batch).squeeze()   # (T,)

    distribution = Categorical(logits = logits)

    actions_t = torch.tensor(
        [int(a) for a in actions],
        dtype = torch.int64,
        device = device,
    )

    new_log_probs = distribution.log_prob(actions_t)
    entropies = distribution.entropy()

    # old_log_probs come from rollouts as lists of per-step tensors that may
    # be 0-dim OR (1,)-shaped - flatten so the ratio math stays strictly
    # per-timestep (shapes (T,) x (T,) - never broadcasts to (T,T)).
    old_log_probs = torch.stack(old_log_probs).reshape(-1).to(device)
    advantages = torch.tensor(
        advantages,
        dtype=torch.float32,
        device=device,
    )
    returns = torch.tensor(
        returns,
        dtype=torch.float32,
        device=device,
    )

    # Probability ratio
    ratio = torch.exp(
        new_log_probs - old_log_probs
    )

    # PPO clipped objective
    L1 = ratio * advantages

    L2 = torch.clamp(
        ratio,
        1 - epsilon,
        1 + epsilon
    ) * advantages

    ppo_objective = torch.minimum(L1, L2)

    # Entropy bonus with an adaptive kick: if the policy is collapsing toward
    # deterministic (mean entropy well below the uniform-random floor) the
    # bonus is amplified up to 4x, which re-injects exploration and stops the
    # premature convergence that previously froze the policy at a local
    # optimum partway through training.
    ent = entropies.mean()
    n_actions = float(logits.shape[-1])
    target_entropy = float(np.log(max(n_actions, 2.0)))

    ent_val = float(ent.detach().cpu())

    if ent_val < 0.6 * target_entropy:
        gap = (target_entropy - ent_val) / target_entropy
        ent_scale = float(np.clip(1.0 + 3.0 * gap, 1.0, 4.0))
    else:
        ent_scale = 1.0

    # Actor loss = negative of the clipped objective, plus a small entropy
    # bonus that rewards a more exploratory (less collapsed) policy.
    actor_loss = (
        -ppo_objective.mean()
        - entropy_coef * ent_scale * ent
    )

    # Critic loss. When old_values are supplied (as in training), use the
    # PPO2-style clipped value loss so the critic can't be destabilized by
    # single-outlier updates at the ends of truncated episodes.
    if old_values is not None:
        old_values_t = torch.tensor(
            old_values,
            dtype = torch.float32,
            device = device,
        )
        values_clipped = old_values_t + torch.clamp(
            values - old_values_t,
            -epsilon,
            epsilon,
        )
        value_loss_unclipped = (values - returns) ** 2
        value_loss_clipped = (values_clipped - returns) ** 2
        critic_loss = (
            torch.maximum(value_loss_unclipped, value_loss_clipped)
        ).mean()
    else:
        critic_loss = (
            (values - returns) ** 2
        ).mean()

    return (
        new_log_probs,
        ratio,
        ppo_objective,
        actor_loss,
        critic_loss,
        ent,
    )