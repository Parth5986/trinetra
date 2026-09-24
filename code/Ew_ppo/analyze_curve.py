import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ckpt = torch.load("checkpoint.pt", map_location="cpu", weights_only=True)
episode = ckpt["episode"]
rew = np.array(ckpt["total_reward_history"], dtype=float)
losses = np.array(ckpt["all_losses"], dtype=float)

print(f"checkpoint episode: {episode}  | seed: {ckpt.get('seed')}")
print(f"reward samples: {len(rew)}   loss samples: {len(losses)}")

def roll(a, w):
    return np.convolve(a, np.ones(w) / w, mode="valid")

print("\n=== rolling mean reward at episode checkpoints ===")
for w, label in ((50, "50-ep"), (100, "100-ep"), (200, "200-ep")):
    r = roll(rew, w)
    n = len(r)
    pts = [int(n * q) for q in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)]
    vals = " | ".join(f"{r[min(p, n-1)]:7.1f}" for p in pts)
    print(f"{label:8s}: {vals}")

print("\n=== final 300 episodes in thirds (is it still rising?) ===")
tail = rew[-300:]
t = len(tail) // 3
for i in range(3):
    seg = tail[i * t:(i + 1) * t]
    print(f"  eps {episode-300+i*t+1:5d}-{episode-300+(i+1)*t:5d}: mean {seg.mean():7.1f}  "
          f"median {np.median(seg):7.1f}  max {seg.max():7.1f}")

# momentum: compare last 150 vs the 150 before that
m_new = rew[-150:].mean()
m_old = rew[-300:-150].mean()
print(f"\nmomentum: mean(last 150)={m_new:.1f} vs mean(150 before)={m_old:.1f} -> delta {m_new-m_old:+.1f}")

# best sustained level: rolling 100 max
r100 = roll(rew, 100)
print(f"best rolling-100 mean: {r100.max():.1f} at episode {np.argmax(r100)+100}")

# plateau ratio: how flat is the last half?
half = roll(rew, 200)
last_200 = half[-200:]
f = np.polyfit(np.arange(len(last_200)), last_200, 1)[0]
print(f"slope of 200-ep rolling mean over its last 200 points: {f:+.3f} reward/episode")
print(f"per-1000-episode trend at current slope: {f*1000:+.1f}")

# loss trend
ep_len = 10
n_batches = len(losses) // ep_len
avg = losses[:n_batches * ep_len].reshape(n_batches, ep_len).mean(axis=1)
print("\n=== batch mean-loss trend ===")
for q in (0.25, 0.5, 0.75, 0.9, 1.0):
    i = int((len(avg) - 1) * q)
    print(f"  ~episode {(i+1)*4:5d}: mean-loss {avg[i]:.4f}")

# figure
fig, axes = plt.subplots(2, 1, figsize=(11, 7))
ax = axes[0]
ax.plot(rew, alpha=0.25, lw=0.5, label="raw per-episode reward")
ax.plot(roll(rew, 50), lw=1.0, label="rolling 50")
ax.plot(roll(rew, 200), lw=1.5, label="rolling 200")
ax.axhline(r100.max(), color="gray", ls="--", lw=1,
           label=f"best rolling-100 ({r100.max():.0f})")
ax.set_title(f"PPO training curve (checkpoint episode {episode})")
ax.set_xlabel("episode"); ax.set_ylabel("episode reward")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(avg, lw=1.0, color="crimson", label="bott mean loss")
ax.set_title("Loss (per multi-episode batch)")
ax.set_xlabel("batch"); ax.set_ylabel("mean loss")
ax.legend(); ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("training_curve.png", dpi=130)
print("\nsaved plot -> training_curve.png")