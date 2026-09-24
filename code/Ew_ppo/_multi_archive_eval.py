import sys
import os
import glob
import re
import argparse

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
ck = torch.load(os.path.join(BASE, "checkpoint.pt"), map_location="cpu", weights_only=True)

actor = Actor(80, 10)
actor.load_state_dict(ck["actor"])
actor.eval()

critic = Critic(80)
critic.load_state_dict(ck["critic"])
critic.eval()

TSRD_DIR = os.path.join(BASE, os.pardir, os.pardir, "data", "tsrd", "archive", "test")
files = sorted(
    glob.glob(os.path.join(TSRD_DIR, "test_*.h5")),
    key=lambda p: int(re.search(r"test_(\d+)\.h5", os.path.basename(p)).group(1)),
)

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--n-files", type=int, default=30, help="how many archives to sample (spread evenly)")
ap.add_argument("--seeds", type=int, default=3, help="resets per file (seeds 1001..)")
ap.add_argument("--device", type=str, default="cuda")
ap.add_argument("--greedy", action="store_true",
                help="use argmax (deterministic) actions instead of sampling")
args = ap.parse_args()

# Evenly spaced file sample across the 151 archives.
idx = np.unique(np.round(np.linspace(0, len(files) - 1, args.n_files)).astype(int))
sample = [files[i] for i in sorted(idx)]

device = torch.device(args.device)
actor = actor.to(device)
critic = critic.to(device)
rows = []

for fp in sample:
    name = os.path.basename(fp)
    for sd in range(1001, 1001 + args.seeds):
        env = EWenv(config=EnvConfig(tsrd_path=fp))
        env.reset(seed=sd)
        obs, acts, rews, dones, olp, vals, term = collect_rollout(
            env, actor, critic, device, greedy=args.greedy
        )
        r = float(np.sum(rews))
        hits = int(np.sum([np.isclose(x, EnvConfig.hit) for x in rews]))
        miss = int(np.sum([np.isclose(x, EnvConfig.miss) for x in rews]))
        fa = int(np.sum([np.isclose(x, EnvConfig.false_alarm) for x in rews]))
        E = hits + miss
        rows.append((name, sd, r, E, hits, fa, 10.5 * E - 500))

hdr = "{:>14} | {:>9} | {:>16} | {:>7} | {:>7} | {:>6} | {:>3}"
print(hdr.format("file", "E min-max", "reward mean+sd", "eff %", "det %", "ceiling", "fa"))
all_rew, all_eff, all_det = [], [], []
reachable = []

for name in sorted(set(r[0] for r in rows)):
    g = [x for x in rows if x[0] == name]
    E = [x[3] for x in g]
    R = [x[2] for x in g]
    H = [x[4] for x in g]
    F = [x[5] for x in g]
    C = [x[6] for x in g]
    eff = [r / c * 100 for r, c in zip(R, C) if c > 0]
    det = [h / e * 100 for h, e in zip(H, E) if e > 0]
    all_rew += R
    all_eff += eff
    all_det += det
    if np.mean(C) >= 7000:
        reachable.append(name)
    eff_s = "{:.1f}".format(np.mean(eff)) if eff else "  -  "
    det_s = "{:.1f}".format(np.mean(det)) if det else "  -  "
    print(hdr.format(
        name,
        "{}-{}".format(min(E), max(E)),
        "{:.1f} +- {:.1f}".format(np.mean(R), np.std(R)),
        eff_s,
        det_s,
        "{:.0f}".format(np.mean(C)),
        "{}-{}".format(min(F), max(F)),
    ))

print()
print("sampled {} archives x {} resets = {} runs".format(len(sample), args.seeds, len(rows)))
degenerate = [r for r in rows if r[3] <= 0]
if all_eff:
    print("ALL runs: mean reward {:.1f} +- {:.1f} | mean eff {:.1f}% | mean det {:.1f}%".format(
        np.mean(all_rew), np.std(all_rew), np.mean(all_eff), np.mean(all_det)))
else:
    print("ALL runs: NO occupied-band scans at all (degenerate rollout)")
if degenerate:
    print("degenerate runs (zero occupied scans): {}".format(
        ", ".join("{} / seed {}".format(r[0], r[1]) for r in degenerate)))
print("archives whose ceiling allows >=7000 (E>=715): {} of {} sampled".format(len(reachable), len(sample)))