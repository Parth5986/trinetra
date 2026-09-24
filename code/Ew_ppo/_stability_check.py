import sys
import os
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

TSRD = os.path.join(BASE, os.pardir, os.pardir, "data", "tsrd", "archive", "test", "test_0.h5")

ap = argparse.ArgumentParser()
ap.add_argument("--greedy", action="store_true",
                help="use argmax (deterministic) actions instead of sampling")
args = ap.parse_args()

print("eval mode:", "greedy (argmax, deterministic)" if args.greedy else "stochastic (sampled policy)")

N = 3
rows = []

for sd in range(1001, 1017):
    for k in range(N):
        env = EWenv(config=EnvConfig(tsrd_path=TSRD))
        env.reset(seed=sd)
        obs, acts, rews, dones, olp, vals, term = collect_rollout(
            env, actor, critic, "cpu", greedy=args.greedy
        )
        r = float(np.sum(rews))
        hits = int(np.sum([np.isclose(x, EnvConfig.hit) for x in rews]))
        miss = int(np.sum([np.isclose(x, EnvConfig.miss) for x in rews]))
        fa = int(np.sum([np.isclose(x, EnvConfig.false_alarm) for x in rews]))
        E = hits + miss
        rows.append((sd, k, r, E, hits, fa, 10.5 * E - 500))

hdr = "{:>5} | {:>11} | {:>16} | {:>16} | {:>6} | {:>6} | {:>5}"
print(hdr.format("seed", "E min-max", "reward mean+sd", "reward min-max", "eff %", "det %", "fa"))
all_rew, all_eff, all_det = [], [], []

for sd in range(1001, 1017):
    g = [x for x in rows if x[0] == sd]
    E = [x[3] for x in g]
    R = [x[2] for x in g]
    H = [x[4] for x in g]
    C = [x[6] for x in g]
    F = [x[5] for x in g]
    eff = [r / c * 100 for r, c in zip(R, C) if c > 0]
    det = [h / e * 100 for h, e in zip(H, E) if e > 0]
    all_rew += R
    all_eff += eff
    all_det += det
    eff_s = "{:.1f}".format(np.mean(eff)) if eff else "  -  "
    det_s = "{:.1f}".format(np.mean(det)) if det else "  -  "
    print(hdr.format(
        sd,
        "{}-{}".format(min(E), max(E)),
        "{:.1f} +- {:.1f}".format(np.mean(R), np.std(R)),
        "{:.1f}-{:.1f}".format(min(R), max(R)),
        eff_s,
        det_s,
        "{}-{}".format(min(F), max(F)),
    ))

degenerate = [r for r in rows if r[3] <= 0]

print()
if all_eff:
    print("ALL {} runs: mean reward {:.1f} +- {:.1f} | mean eff {:.1f}% of ceiling | mean det {:.1f}%".format(
        len(rows), np.mean(all_rew), np.std(all_rew), np.mean(all_eff), np.mean(all_det)))
else:
    print("ALL runs: NO occupied-band scans at all (degenerate greedy rollout)")
if degenerate:
    print("degenerate runs (zero occupied scans): {}".format(
        ", ".join("seed {}/reset {}".format(r[0], r[1]) for r in degenerate)))
w = np.argsort(all_rew)
print("worst 4 (seed/reset -> reward, det, E):")
for i in w[:4]:
    g = [x for x in rows if x[2] == all_rew[i]]
    sd, k, r, E, hits, fa, c = g[0]
    dstr = "{:.1f}%".format(hits / E * 100) if E > 0 else "   n/a "
    print("  {} / reset {} -> {:.1f} | det {} | E {}".format(sd, k, r, dstr, E))
print("best 4:")
for i in w[-4:][::-1]:
    g = [x for x in rows if x[2] == all_rew[i]]
    sd, k, r, E, hits, fa, c = g[0]
    dstr = "{:.1f}%".format(hits / E * 100) if E > 0 else "   n/a "
    print("  {} / reset {} -> {:.1f} | det {} | E {}".format(sd, k, r, dstr, E))