import argparse
import sys
import os
import glob
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "trinetra_env"))

import numpy as np
import torch

from agent import Actor, Critic
from rollout import collect_rollout
from config import EnvConfig
from env import EWenv

BASE = os.path.dirname(os.path.abspath(__file__))
TSRD_DIR = os.path.join(BASE, os.pardir, os.pardir, "data", "tsrd", "archive", "test")

DEFAULT_BASELINE = "checkpoint_OLD_baseline.pt"
DEFAULT_NEW = "checkpoint.pt"


def resolve_ckpt(path: str) -> str:
    """Bare file names resolve against the PPO dir; explicit/absolute paths pass through."""
    if os.path.exists(path):
        return path
    cand = os.path.join(BASE, path)
    if not os.path.exists(cand):
        raise SystemExit(f"checkpoint not found: {path} (also tried {cand})")
    return cand


def load_policy(ck_path, device):
    ck = torch.load(ck_path, map_location="cpu", weights_only=True)
    if "actor" not in ck or "critic" not in ck:
        raise SystemExit(
            f"{ck_path} is not a full training checkpoint (missing actor/critic keys). "
            "Use a full checkpoint such as checkpoint.pt / best_checkpoint.pt, "
            "not an actor-only export."
        )
    actor = Actor(80, 10)
    critic = Critic(80)
    actor.load_state_dict(ck["actor"])
    critic.load_state_dict(ck["critic"])
    actor.eval().to(device)
    critic.eval().to(device)
    return actor, critic, ck.get("episode")


def evaluate(name, ck_path, greedy, sample_files, seeds, device):
    actor, critic, ep = load_policy(ck_path, device)

    all_r, all_hits, all_fa, all_miss = [], [], [], []
    for fp in sample_files:
        for sd in seeds:
            env = EWenv(config=EnvConfig(tsrd_path=fp))
            env.reset(seed=sd)
            obs, acts, rews, dones, olp, vals, term = collect_rollout(
                env, actor, critic, device, greedy=greedy
            )
            r = float(np.sum(rews))
            hits = int(np.sum([np.isclose(x, EnvConfig.hit) for x in rews]))
            miss = int(np.sum([np.isclose(x, EnvConfig.miss) for x in rews]))
            fa = int(np.sum([np.isclose(x, EnvConfig.false_alarm) for x in rews]))
            all_r.append(r); all_hits.append(hits); all_fa.append(fa); all_miss.append(miss)
            env.close()

    mode = "greedy" if greedy else "sampled"
    print(f"\n=== {name} (trained to ep {ep}, {mode}) | {len(all_r)} episodes ===")
    print(f"  reward      mean {np.mean(all_r):8.1f}   median {np.median(all_r):8.1f}   std {np.std(all_r):7.1f}")
    print(f"  detections  mean {np.mean(all_hits):7.1f}   total {sum(all_hits):6d}")
    print(f"  misses      mean {np.mean(all_miss):7.1f}")
    print(f"  false-alarm mean {np.mean(all_fa):6.2f}   total {sum(all_fa):5d}")
    return np.mean(all_r)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and compare PPO checkpoints on held-out TSRD episodes"
    )
    parser.add_argument(
        "--ckpt", nargs="+", default=[DEFAULT_NEW],
        help="checkpoints to evaluate (paths or bare names in the PPO dir). default: %(default)s",
    )
    parser.add_argument(
        "--baseline", default=DEFAULT_BASELINE,
        help="reference policy for the delta report. default: %(default)s",
    )
    parser.add_argument(
        "--sample", type=int, default=80,
        help="number of archive files to sample (default 80)",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[1001, 1002, 1003],
        help="evaluation seeds per file (default: 1001 1002 1003)",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="torch device (default cuda)",
    )
    parser.add_argument(
        "--mode", choices=["both", "greedy", "sampled"], default="both",
        help="evaluation modes (default both)",
    )
    args = parser.parse_args()

    files = sorted(
        glob.glob(os.path.join(TSRD_DIR, "test_*.h5")),
        key=lambda p: int(re.search(r"test_(\d+)\.h5", os.path.basename(p)).group(1)),
    )
    if not files:
        raise SystemExit(f"no test_*.h5 files found in {TSRD_DIR}")

    idx = np.unique(np.round(np.linspace(0, len(files) - 1, args.sample)).astype(int))
    sample = [files[i] for i in sorted(idx)]

    baseline_path = resolve_ckpt(args.baseline)
    ckpt_paths = [resolve_ckpt(p) for p in args.ckpt]

    device = torch.device(args.device)

    def mode_flags():
        if args.mode == "greedy":
            return (True,)
        if args.mode == "sampled":
            return (False,)
        return (True, False)

    print(f"files available: {len(files)} | sample: {len(sample)} | "
          f"seeds: {args.seeds} | device: {args.device}")

    anchors = [baseline_path] + [p for p in ckpt_paths if p != baseline_path]

    results = {}
    for path in anchors:
        name = os.path.splitext(os.path.basename(path))[0]
        for greedy in mode_flags():
            r_mean = evaluate(name, path, greedy, sample, args.seeds, device)
            results[(name, greedy)] = r_mean

    base_name = os.path.splitext(os.path.basename(baseline_path))[0]
    for path in [p for p in ckpt_paths if p != baseline_path]:
        name = os.path.splitext(os.path.basename(path))[0]
        for greedy in mode_flags():
            mode = "greedy" if greedy else "sampled"
            new_r = results[(name, greedy)]
            old_r = results[(base_name, greedy)]
            pct = (new_r - old_r) / abs(old_r) * 100 if old_r else float("nan")
            print(f"\n  -> delta ({mode}): {name} vs {base_name}: "
                  f"{new_r - old_r:+.1f} mean reward ({pct:+.1f}%)")


if __name__ == "__main__":
    main()