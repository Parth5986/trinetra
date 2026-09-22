import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))

ENV_DIR = r"D:\trinetra\trinetra_env"

if ENV_DIR not in sys.path:
    sys.path.insert(0, ENV_DIR)

from agent import Actor, Critic
from rollout import collect_rollout
from ppo import ppo

from config import EnvConfig

EnvConfig.tsrd_path = str(
    (
        Path(
            r"D:\trinetra\data\tsrd\archive\test\test_0.h5"
        )
    ).resolve()
)

from gae import calculate_gae, calculate_returns

from env import EWenv


def main():

    parser = argparse.ArgumentParser(description = "PPO trainer for the EW environment")

    parser.add_argument("--seed", type = int, default = 0)

    parser.add_argument("--episodes", type = int, default = 100)

    parser.add_argument("--epochs", type = int, default = 40)

    parser.add_argument("--epsilon", type = float, default = 0.2)

    parser.add_argument("--gamma", type = float, default = 0.99)

    parser.add_argument("--lam", type = float, default = 0.95)

    parser.add_argument("--lr", type = float, default = 3e-4)

    parser.add_argument("--device", type = str, default = "cpu")

    parser.add_argument(
        "--restart",
        action = "store_true",
        help = "ignore checkpoint.pt and (re)train from scratch",
    )

    parser.add_argument(
        "--reset-counter",
        action = "store_true",
        help = (
            "load trained actor/critic/optimizer weights from checkpoint.pt "
            "but restart the episode counter at 1 (trained policy kept, "
            "episode numbering restarts)"
        ),
    )

    args = parser.parse_args()

    CHECKPOINT_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "checkpoint.pt",
    )

    torch.manual_seed(args.seed)

    np.random.seed(args.seed)

    device = torch.device(args.device)

    obs_size = 10 * 8

    action_size = 10

    env = EWenv()

    actor = Actor(obs_size, action_size).to(device)

    critic = Critic(obs_size).to(device)

    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr = args.lr,
    )

    total_reward_history = []

    all_losses = []

    # Checkpoint semantics:
    # - By default we RESUME: pick up actor/critic/optimizer + the episode
    #   counter + loss history from checkpoint.pt if it exists, and keep
    #   training from the next episode instead of starting over.
    # - --restart ignores the checkpoint and starts fresh (also overwrites
    #   it on the first save of the new run).

    start_episode = 1

    if os.path.exists(CHECKPOINT_PATH) and not args.restart:

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location = device,
            weights_only = True,
        )

        actor.load_state_dict(checkpoint["actor"])

        critic.load_state_dict(checkpoint["critic"])

        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])

        if args.reset_counter:
            start_episode = 1
            total_reward_history = []
            all_losses = []
            print(
                "loaded trained weights; episode counter reset to 1 "
                "(continuing with trained policy, blocks restart at 1)"
            )
        else:
            start_episode = int(checkpoint["episode"]) + 1
            total_reward_history = list(checkpoint["total_reward_history"])
            all_losses = list(checkpoint["all_losses"])
            print(
                f"resumed from checkpoint.pt (continuing at episode {start_episode})"
            )

    elif not args.restart:

        print("no checkpoint found - starting fresh")

    else:

        print("--restart set - ignoring checkpoint, training from scratch")

    for episode in range(start_episode, args.episodes + 1):

        # Seed the environment so each episode is reproducible and each
        # K-followup voxel draws from the seeded RNG (not the global one).

        env.reset(seed = args.seed + episode)

        rollout = collect_rollout(
            env,
            actor,
            critic,
            device,
        )

        (
        observations,
        actions,
        rewards,
        dones,
        old_log_probs,
        values,
        ) = rollout

        advantages = calculate_gae(
            rewards,
            values,
            dones,
            args.lam,
            args.gamma,
        )

        returns = calculate_returns(
            advantages,
            values,
        )

        total_episode_reward = float(np.sum(rewards))

        # Per-episode metrics decoded from the unique reward constants
        # (config.py hit=+10 miss=-5 false_alarm=-3 empty=-0.5).

        episode_detections = int(np.sum(
            [np.isclose(r, EnvConfig.hit) for r in rewards]
        ))

        episode_misses = int(np.sum(
            [np.isclose(r, EnvConfig.miss) for r in rewards]
        ))

        episode_false_alarms = int(np.sum(
            [np.isclose(r, EnvConfig.false_alarm) for r in rewards]
        ))

        # PPO per-episode updates

        episode_losses = []

        for epoch in range(args.epochs):

            (
            new_log_probs,
            ratio,
            ppo_objective,
            actor_loss,
            critic_loss,
            ) = ppo(
                actor,
                critic,
                observations,
                actions,
                old_log_probs,
                advantages,
                returns,
                args.epsilon,
            )

            loss = actor_loss + 0.5 * critic_loss

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(critic.parameters()),
                max_norm = 0.5,
            )

            optimizer.step()

            episode_losses.append(float(loss.detach()))

        all_losses.extend(episode_losses)

        if episode % 10 == 0 or episode == 1:

            print(
                f"episode {episode:5d} | "
                f"episode-reward {float(total_episode_reward):8.1f} | "
                f"detections {episode_detections:3d} | "
                f"false-alarms {episode_false_alarms:3d} | "
                f"missed-emitters {episode_misses:3d} | "
                f"steps {len(rewards):4d} | "
                f"mean-loss {float(np.mean(episode_losses)):.6f}"
            )

        # Persist full training state so the next invocation resumes from
        # where this run left off instead of training from scratch.

        torch.save(
            {
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "optimizer": optimizer.state_dict(),
                "episode": episode,
                "total_reward_history": total_reward_history,
                "all_losses": all_losses,
                "seed": args.seed,
            },
            CHECKPOINT_PATH,
        )

        # The dashboard reads PPO_DIR/actor.pt (not checkpoint.pt), so no
        # amount of training reaches the PPO row until the frozen actor is
        # exported separately next to the checkpoint.
        torch.save(
            actor.state_dict(),
            os.path.join(os.path.dirname(CHECKPOINT_PATH), "actor.pt"),
        )

    print("-" * 50)

    print(
        f"total episodes: {args.episodes} | "
        f"final mean loss: {float(np.mean(all_losses[-args.epochs:])):.6f}"
    )


if __name__ == "__main__":
    main()
