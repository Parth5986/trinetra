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
    epsilon
):
    new_log_probs = []
    new_values = []

    for i in range(len(observations)):

        obs_tensor = torch.tensor(
            observations[i],
            dtype = torch.float32
        ).reshape(1, -1)

        logits = actor(obs_tensor)
        value = critic(obs_tensor)

        distribution = Categorical(logits=logits)

        new_log_prob = distribution.log_prob(actions[i])

        new_log_probs.append(new_log_prob)
        new_values.append(value.squeeze())

    new_log_probs = torch.stack(new_log_probs)
    new_values = torch.stack(new_values)

    old_log_probs = torch.stack(old_log_probs)
    advantages = torch.tensor(
        advantages,
        dtype=torch.float32
    )
    returns = torch.tensor(
        returns,
        dtype=torch.float32
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

    # Actor loss
    actor_loss = -ppo_objective.mean()

    # Critic loss
    critic_loss = (
        (new_values - returns) ** 2
    ).mean()

    return (
        new_log_probs,
        ratio,
        ppo_objective,
        actor_loss,
        critic_loss
    )