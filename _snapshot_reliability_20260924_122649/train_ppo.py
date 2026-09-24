import argparse
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))

PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).resolve().parents[1]
ENV_DIR = str(PROJECT_ROOT / "trinetra_env")

if ENV_DIR not in sys.path:
    sys.path.insert(0, ENV_DIR)

from agent import Actor, Critic
from rollout import collect_rollout, collect_rollouts
from ppo import ppo

from config import EnvConfig
from gae import calculate_gae, calculate_returns

from env import EWenv

TSRD_ARCHIVE_DIR = str(
    (PROJECT_ROOT / "data" / "tsrd" / "archive" / "test").resolve()
)


def resolve_archives(archives_arg: str) -> list:
    """Turn the --archives argument into a concrete, ordered list of h5 paths.

    "all"          -> every test_*.h5 in the TSRD archive test dir (numerical
                      order, so episode rotation is deterministic).
    otherwise      -> comma-separated file names (resolved against the archive
                      dir) or explicit paths, in the given order.
    """
    if not archives_arg:
        raise SystemExit("--archives must not be empty ('all' = every file)")

    if archives_arg == "all":
        files = sorted(
            glob.glob(os.path.join(TSRD_ARCHIVE_DIR, "test_*.h5")),
            key = lambda p: int(
                re.search(r"test_(\d+)\.h5", os.path.basename(p)).group(1)
            ),
        )
        if not files:
            raise SystemExit(f"no test_*.h5 files found in {TSRD_ARCHIVE_DIR}")
        return files

    paths = []
    for name in [s.strip() for s in archives_arg.split(",") if s.strip()]:
        cand = name if os.path.isabs(name) else os.path.join(TSRD_ARCHIVE_DIR, name)
        if not os.path.exists(cand):
            raise SystemExit(f"--archives file not found: {cand}")
        paths.append(cand)
    if not paths:
        raise SystemExit(
            "--archives: no files parsed (use 'all' or comma-separated names)"
        )
    return paths


def main():

    parser = argparse.ArgumentParser(description = "PPO trainer for the EW environment")

    parser.add_argument("--seed", type = int, default = 0)

    parser.add_argument("--episodes", type = int, default = 100)

    parser.add_argument("--epochs", type = int, default = 40)

    parser.add_argument(
        "--minibatch",
        type = int,
        default = None,
        help = (
            "minibatch size in timesteps for each epoch's PPO update. "
            "default None = whole combined batch per epoch (original "
            "behaviour). With --n-envs>1, a minibatch of a few hundred to "
            "~1k timesteps and a smaller --epochs (3-5) is the stable regime."
        ),
    )

    parser.add_argument("--n-envs", type = int, default = 1,
        help = (
            "collect n_envs episodes per PPO update (parallel environments); "
            "default 1 keeps the original one-episode-per-update behaviour"
        ))

    parser.add_argument(
        "--archives",
        type = str,
        default = "all",
        help = (
            "TSRD test files to train on: 'all' = every test_*.h5 in "
            "data/tsrd/archive/test (default), or a comma-separated list "
            "such as 'test_0.h5,test_2.h5' (names resolved against the "
            "archive dir) or explicit paths. Episodes rotate round-robin "
            "across the list."
        ),
    )

    parser.add_argument("--epsilon", type = float, default = 0.2)

    parser.add_argument("--entropy-coef", type = float, default = 0.1,
        help = (
            "entropy bonus weight in the actor loss (default 0.1); "
            "set 0.0 to disable, larger values keep the policy exploring"
        ))

    parser.add_argument("--entropy-end", type = float, default = 0.05,
        help = (
            "if set, anneal the entropy coefficient linearly from "
            "--entropy-coef at the start of this run down to --entropy-end "
            "at --episodes (explore early, keep a floor so the policy "
            "doesn't collapse to deterministic). default: 0.05"
        ))

    parser.add_argument("--gamma", type = float, default = 0.99)

    parser.add_argument("--lam", type = float, default = 0.95)

    parser.add_argument("--lr", type = float, default = 3e-4)

    parser.add_argument("--lr-end", type = float, default = None,
        help = (
            "if set, anneal the learning rate linearly from --lr at the "
            "start of this run down to --lr-end at --episodes. default: no "
            "annealing"
        ))

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

    parser.add_argument(
        "--log-every",
        type = int,
        default = 10,
        help = (
            "print per-epoch loss details every N episodes so you can watch "
            "the loss decrease across epochs/episodes (default 10)"
        ),
    )

    parser.add_argument(
        "--best-window",
        type = int,
        default = 100,
        help = (
            "rolling window (episodes) of mean reward used to decide when a "
            "new best checkpoint is saved. The reward is noisy and the curve "
            "peaks mid-run, so this keeps the peak policy instead of the last "
            "one (default 100)"
        ),
    )

    parser.add_argument(
        "--best-margin",
        type = float,
        default = 0.01,
        help = (
            "relative improvement required to overwrite best_checkpoint.pt / "
            "best_actor.pt (default 0.01 = +1%%)"
        ),
    )

    args = parser.parse_args()

    if args.n_envs < 1:
        print("--n-envs must be >= 1 (default 1)")
        return

    if args.minibatch is not None and args.minibatch < 1:
        args.minibatch = None
        print("--minibatch < 1 ignored - using the full batch per epoch")

    archives = resolve_archives(args.archives)

    print(
        f"training over {len(archives)} TSRD archive(s): "
        + ", ".join(os.path.basename(a) for a in archives[:5])
        + (" ..." if len(archives) > 5 else "")
    )

    CHECKPOINT_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "checkpoint.pt",
    )

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")

    np.random.seed(args.seed)

    device = torch.device(args.device)

    obs_size = 10 * 8

    action_size = 10

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

    # Tracks the mean loss of the last logged episode block so we can print
    # whether the loss is going down (learning) between blocks.

    last_logged_mean_loss = None

    # Best-checkpoint tracking: rolling-mean reward of the best policy seen so
    # far, and the episode at which it occurred (for the final report).
    best_roll = -float("inf")
    best_saved_at = None

    # Rolling windows for reward / loss so the trend can be seen over noise.
    # Losses are kept per component (actor / critic / total) so the critic's
    # reward-scale swings don't hide what the policy is doing.

    recent_rewards = []
    recent_losses = []
    recent_actor_losses = []
    recent_critic_losses = []

    episode = start_episode

    while episode <= args.episodes:

        k_eff = min(args.n_envs, args.episodes - episode + 1)

        # Rebuild the environments for this iteration: EWenv binds its TSRD
        # file in __init__, so each env gets a fresh instance whose scenario
        # file rotates round-robin over the archive list. The assignment is a
        # deterministic function of the episode number, so resuming mid-run
        # keeps the same per-episode scenario file.

        envs = [
            EWenv(
                config = EnvConfig(
                    tsrd_path = archives[(episode - 1 + j) % len(archives)]
                )
            )
            for j in range(k_eff)
        ]

        archive_files = [
            os.path.basename(archives[(episode - 1 + j) % len(archives)])
            for j in range(k_eff)
        ]

        # Seed each environment with a distinct, reproducible seed. The
        # numbering extends the sequential mapping used by n_envs=1, so
        # resuming with a different --n-envs stays consistent episode by
        # episode.

        for j in range(k_eff):
            envs[j].reset(seed = args.seed + episode + j)

        # Anneal entropy + learning rate over this run's span
        # (start_episode .. args.episodes). Linear schedule: start at the
        # configured values and glide to --entropy-end / --lr-end, so the
        # policy explores early and sharpens late (helps the argmax converge).

        run_span = max(args.episodes - start_episode, 1)
        anneal_frac = (episode - start_episode) / run_span

        if args.entropy_end is not None:
            entropy_coef_now = (
                args.entropy_coef
                + (args.entropy_end - args.entropy_coef) * anneal_frac
            )
        else:
            entropy_coef_now = args.entropy_coef

        if args.lr_end is not None:
            lr_now = args.lr + (args.lr_end - args.lr) * anneal_frac
            for g in optimizer.param_groups:
                g["lr"] = lr_now
        else:
            lr_now = float(optimizer.param_groups[0]["lr"])

        if k_eff == 1:
            rollouts = [collect_rollout(envs[0], actor, critic, device)]
        else:
            rollouts = collect_rollouts(envs[:k_eff], actor, critic, device)

        combined_observations = []
        combined_actions = []
        combined_old_log_probs = []
        combined_old_values = []
        combined_advantages = []
        combined_returns = []

        per_episode = []

        for j, rollout in enumerate(rollouts):

            ep_no = episode + j

            (
            observations,
            actions,
            rewards,
            dones,
            old_log_probs,
            values,
            terminals,
            ) = rollout

            # Critic values live on the training device (GPU) during the
            # rollout. Convert them to plain floats first so the GAE
            # bookkeeping stays device-independent; ppo() then re-uploads
            # them where needed.

            values_cpu = [float(v.detach().cpu()) for v in values]

            advantages = calculate_gae(
                rewards,
                values_cpu,
                dones,
                args.lam,
                args.gamma,
                terminals = terminals,
            )

            # Standard PPO trick: whiten the advantages so the actor's
            # updates no longer depend on the raw reward scale, and the
            # critic learns on a stable unit-variance target.

            adv_arr = np.asarray(advantages, dtype = np.float32)
            if adv_arr.std() > 1e-6:
                adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)
            advantages = adv_arr.tolist()

            returns = calculate_returns(
                advantages,
                values_cpu,
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

            total_reward_history.append(total_episode_reward)
            recent_rewards.append(total_episode_reward)

            combined_observations.extend(observations)
            combined_actions.extend(actions)
            combined_old_log_probs.extend(old_log_probs)
            combined_old_values.extend(values_cpu)
            combined_advantages.extend(advantages)
            combined_returns.extend(returns)

            per_episode.append(
                {
                    "ep_no": ep_no,
                    "file": archive_files[j],
                    "reward": total_episode_reward,
                    "detections": episode_detections,
                    "false_alarms": episode_false_alarms,
                    "missed": episode_misses,
                    "steps": len(rewards),
                }
            )

        # PPO per-epoch updates over the combined batch - one update per
        # epoch regardless of how many rollouts this iteration collected.

        episode_losses = []
        episode_actor_losses = []
        episode_critic_losses = []

        n_steps = len(combined_observations)

        # Minibatched PPO: each epoch reshuffles the combined batch and takes
        # one gradient step per minibatch. With n_envs>1 the minibatches span
        # independent (parallel) episodes, which is what makes the shuffle
        # effective. When --minibatch is unset, the "minibatch" is simply the
        # whole batch, preserving the original one-update-per-epoch behaviour.

        minibatch = n_steps if args.minibatch is None else min(n_steps, args.minibatch)

        for epoch in range(args.epochs):

            epoch_losses = []
            epoch_actor_losses = []
            epoch_critic_losses = []

            perm = np.random.permutation(n_steps)

            for start in range(0, n_steps, minibatch):

                idx = perm[start:start + minibatch]

                mb_observations = [combined_observations[j] for j in idx]
                mb_actions = [combined_actions[j] for j in idx]
                mb_old_log_probs = [combined_old_log_probs[j] for j in idx]
                mb_advantages = [combined_advantages[j] for j in idx]
                mb_returns = [combined_returns[j] for j in idx]
                mb_old_values = [combined_old_values[j] for j in idx]

                (
                new_log_probs,
                ratio,
                ppo_objective,
                actor_loss,
                critic_loss,
                entropy_now,
                ) = ppo(
                    actor,
                    critic,
                    mb_observations,
                    mb_actions,
                    mb_old_log_probs,
                    mb_advantages,
                    mb_returns,
                    args.epsilon,
                    device,
                    entropy_coef_now,
                    old_values = mb_old_values,
                )

                loss = actor_loss + 0.5 * critic_loss

                optimizer.zero_grad()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()),
                    max_norm = 0.5,
                )

                optimizer.step()

                epoch_losses.append(float(loss.detach()))
                epoch_actor_losses.append(float(actor_loss.detach()))
                epoch_critic_losses.append(float(critic_loss.detach()))

            # One aggregated entry per epoch (mean over its minibatches) so
            # the per-epoch logging and trend code below stay unchanged.
            episode_losses.append(float(np.mean(epoch_losses)))
            episode_actor_losses.append(float(np.mean(epoch_actor_losses)))
            episode_critic_losses.append(float(np.mean(epoch_critic_losses)))

        all_losses.extend(episode_losses)

        mean_loss = float(np.mean(episode_losses))
        mean_actor_loss = float(np.mean(episode_actor_losses))
        mean_critic_loss = float(np.mean(episode_critic_losses))

        # Keep the rolling loss window aligned with the rolling reward window
        # (one entry per completed episode).

        recent_losses.extend([mean_loss] * k_eff)
        recent_actor_losses.extend([mean_actor_loss] * k_eff)
        recent_critic_losses.extend([mean_critic_loss] * k_eff)

        # Detailed loss logging every --log-every episodes so you can see
        # whether the model is learning.

        if k_eff == 1:
            should_log = (episode % args.log_every == 0 or episode == 1)
        else:
            should_log = (episode == 1 or (episode - 1) % args.log_every == 0)

        if should_log:

            window = min(len(recent_rewards), 10)

            for m in per_episode:
                print(
                    f"episode {m['ep_no']:5d} | "
                    f"file {m['file']} | "
                    f"episode-reward {float(m['reward']):8.1f} | "
                    f"detections {m['detections']:3d} | "
                    f"false-alarms {m['false_alarms']:3d} | "
                    f"missed-emitters {m['missed']:3d} | "
                    f"steps {m['steps']:4d} | "
                    f"mean-loss {mean_loss:.6f} | "
                    f"ent {entropy_coef_now:.4f} | lr {lr_now:.2e}"
                )

            # Rolling trend: averages over the last episodes smooth out the
            # per-episode noise and show whether learning is really moving.
            # Actor/critic are shown separately because the critic loss floats
            # on the batch's reward scale - a noisy critic number is expected
            # and does NOT mean the policy is misbehaving.
            print(
                f"  rolling (last {window} eps): "
                f"avg-reward {float(np.mean(recent_rewards[-window:])):8.1f} | "
                f"avg-loss {float(np.mean(recent_losses[-window:])):.4f} | "
                f"avg-actor {float(np.mean(recent_actor_losses[-window:])):.4f} | "
                f"avg-critic {float(np.mean(recent_critic_losses[-window:])):.4f}"
            )

            win_mean = float(np.mean(recent_losses[-window:]))

            if last_logged_mean_loss is None:
                print(
                    "  loss trend: first sample - compare against the next "
                    "--log-every block"
                )
            else:
                delta = win_mean - last_logged_mean_loss
                trend = "down (learning)" if delta < 0 else "up"
                print(
                    f"  loss trend: {delta:+.6f} vs rolling avg at last "
                    f"logged block ({trend})"
                )

            print("  per-epoch losses (actor | critic | total):")
            for e, (al, cl, tl) in enumerate(
                zip(episode_actor_losses, episode_critic_losses, episode_losses)
            ):
                print(
                    f"    epoch {e + 1:3d}/{args.epochs:3d} | "
                    f"actor {al:9.4f} | critic {cl:10.2f} | "
                    f"total {tl:9.4f}"
                )

            last_logged_mean_loss = win_mean

        # Persist full training state so the next invocation resumes from
        # where this run left off instead of training from scratch.

        torch.save(
            {
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "optimizer": optimizer.state_dict(),
                "episode": episode + k_eff - 1,
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

        # Best-checkpoint tracking: save a SEPARATE best_checkpoint.pt /
        # best_actor.pt whenever the recent rolling-mean reward beats the
        # previous best by at least --best-margin. The reward is noisy and
        # this curve peaks mid-run, so the last save is rarely the best
        # policy - these files keep the actual peak for evaluation/deployment.
        if len(total_reward_history) >= args.best_window:
            roll_now = float(np.mean(total_reward_history[-args.best_window:]))
        else:
            roll_now = float(np.mean(total_reward_history))

        if roll_now > best_roll * (1.0 + max(args.best_margin, 0.0)):
            best_roll = roll_now
            best_saved_at = episode + k_eff - 1
            best_state = {
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "optimizer": optimizer.state_dict(),
                "episode": best_saved_at,
                "total_reward_history": total_reward_history,
                "all_losses": all_losses,
                "seed": args.seed,
            }
            torch.save(
                best_state,
                os.path.join(os.path.dirname(CHECKPOINT_PATH), "best_checkpoint.pt"),
            )
            torch.save(
                actor.state_dict(),
                os.path.join(os.path.dirname(CHECKPOINT_PATH), "best_actor.pt"),
            )
            print(
                f"  ** new best: rolling-{args.best_window} mean {roll_now:.1f} "
                f"at episode {best_saved_at} -> saved best_checkpoint.pt / best_actor.pt"
            )

        episode += k_eff

    print("-" * 50)

    if best_saved_at is not None:
        print(
            f"best checkpoint: rolling-{args.best_window} mean {best_roll:.1f} "
            f"at episode {best_saved_at} -> best_checkpoint.pt / best_actor.pt"
        )

    print(
        f"total episodes: {args.episodes} | "
        f"final mean loss: {float(np.mean(all_losses[-args.epochs:])):.6f}"
    )


if __name__ == "__main__":
    main()