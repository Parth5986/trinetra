import sys, types, numpy as np

fake_gym = types.ModuleType("gymnasium")
fr = types.ModuleType("gymnasium.spaces")

class Discrete:
    def __init__(s, n): s.n = n
    def contains(s, x): return 0 <= int(x) < s.n
class Box:
    def __init__(s, low, high, dtype): s.low, s.high, s.dtype = np.asarray(low), np.asarray(high), dtype
class EnvBase:
    metadata = {}
    def reset(s, *, seed=None, options=None): s.np_random = np.random.default_rng(seed); return None
    def step(s, a): return None, 0.0, False, False, {}

fr.Discrete = Discrete; fr.Box = Box
fake_gym.spaces = fr; fake_gym.Env = EnvBase
sys.modules["gymnasium"] = fake_gym
sys.modules["gymnasium.spaces"] = fr

sys.path.insert(0, r"D:\trinetra\trinetra_env")
sys.path.insert(0, r"D:\trinetra\trinetra_env")

import env as E

env = E.EWenv()
obs, info = env.reset(seed=7)

print("obs:", obs.shape, obs.dtype)
print("obs_space:", env.observation_space)
print("act_space:", env.action_space)

def inside_box(box, a):
    return bool(np.all(np.asarray(a) >= box.low) and np.all(np.asarray(a) <= box.high))

ok_obs = inside_box(env.observation_space, obs)
print("obs inside Box:", ok_obs)
print("obs_space.low==high sanity -> low==0 high==? first row:", env.observation_space.high[0])
print("space dtype vs obs dtype:", env.observation_space.dtype, obs.dtype)

rewards = []
all_ok = True
rng = np.random.default_rng(0)
for i in range(50):
    a = int(rng.integers(env.config.n_bands))
    o2, r, term, trunc, inf = env.step(a)
    rewards.append(float(r))
    all_ok = all_ok and inside_box(env.observation_space, o2)

print("50 steps: obs always in space:", all_ok)
print("rewards finite:", np.isfinite(rewards).all(), "| range:", min(rewards), "to", max(rewards))

# determinism of reset
env2 = E.EWenv()
o1, _ = env.reset(seed=123)
o2, _ = env2.reset(seed=123)
print("reset(same seed) deterministic:", np.array_equal(o1, o2))
o3, _ = env2.reset(seed=999)
print("reset(diff seed) differs:", not np.array_equal(o2, o3))
