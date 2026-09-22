import os
import sys

os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

ENV_DIR = r"D:\trinetra\trinetra_env"
PPO_DIR = r"D:\trinetra\code\Ew_ppo"

for p in (ENV_DIR, PPO_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# Environment config must be pinned BEFORE the env module is imported:
# EWenv's TsrdLoader reads EnvConfig.tsrd_path at import time, so a
# relative default would resolve against the dashboard's cwd instead of
# the real h5 archive.

from config import EnvConfig

EnvConfig.tsrd_path = (
    r"D:\trinetra\data\tsrd\archive\test\test_0.h5"
)

import numpy as np
import torch
import streamlit as st
from torch.distributions import Categorical

from env import EWenv
from agent import Actor

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


REWARDS = {
    "hit": +10.0,
    "miss": -5.0,
    "false_alarm": -3.0,
    "empty_band": -0.5,
}


def run_agent(agent, env, n_episodes, seed, results):
    rows = []
    for episode in range(1, n_episodes + 1):
        agent.reset(seed = seed + episode)
        obs, info = env.reset(seed = seed + episode)
        ep_rewards = []
        done = False
        steps = 0
        while not done:
            action = agent.decide(obs, {"step": steps + 1})
            obs, reward, terminated, truncated, info = env.step(action)
            ep_rewards.append(float(reward))
            agent.update(action, reward)
            steps += 1
            done = terminated or truncated
        rows.append(decode_episode(ep_rewards))
    return rows


def decode_episode(rewards):
    return {
        "detections": int(np.sum([np.isclose(r, REWARDS["hit"]) for r in rewards])),
        "false_alarms": int(np.sum([np.isclose(r, REWARDS["false_alarm"]) for r in rewards])),
        "misses": int(np.sum([np.isclose(r, REWARDS["miss"]) for r in rewards])),
        "empties": int(np.sum([np.isclose(r, REWARDS["empty_band"]) for r in rewards])),
        "steps": int(len(rewards)),
        "reward": float(np.sum(rewards)),
    }


class PPO_Frozen:

    # Load the trained actor.pt (EWenv obs 80 -> 256 -> 256 -> 10 bands)
    # once, then sample band actions from its Categorical logits. No
    # gradients, no optimizer - "deployment mode" forward pass only.

    def __init__(self, obs_size = 80, n_bands = 10):
        self.n_bands = n_bands
        self.actor = Actor(obs_size, n_bands)
        self.actor.load_state_dict(
            torch.load(
                os.path.join(PPO_DIR, "actor.pt"),
                map_location = "cpu",
                weights_only = True,
            )
        )
        self.actor.eval()

    def reset(self, seed):
        pass

    def decide(self, obs, info):
        obs_t = torch.as_tensor(obs, dtype = torch.float32).reshape(1, -1)
        with torch.no_grad():
            logits = self.actor(obs_t)
            return int(Categorical(logits = logits).sample().item())

    def update(self, action, reward):
        pass


class RandomAgent:
    def __init__(self, n_bands, seed = 0):
        self.n_bands = n_bands
        self.rng = np.random.default_rng(seed)
    def reset(self, seed):
        self.rng = np.random.default_rng(seed)
    def decide(self, obs, info):
        return int(self.rng.integers(0, self.n_bands))
    def update(self, action, reward):
        pass


class EnergyAgent:
    # Operator heuristic: always probe the highest-energy band.
    def __init__(self, n_bands, seed = 0):
        self.n_bands = n_bands
    def reset(self, seed):
        pass
    def decide(self, obs, info):
        return int(np.argmax(obs[0]))
    def update(self, action, reward):
        pass


class UCB1Agent:
    def __init__(self, n_bands, seed = 0):
        self.n_bands = n_bands
        self.reset(seed)
    def reset(self, seed):
        self.counts = np.ones(self.n_bands)
        self.hits = np.zeros(self.n_bands)
        self.hit_rate = np.zeros(self.n_bands)
        self.t = 0
    def decide(self, obs, info):
        self.t += 1
        if self.t <= self.n_bands:
            return int((self.t - 1) % self.n_bands)
        log_t = np.log(float(self.t))
        bonus = np.sqrt(2.0 * log_t / np.maximum(self.counts, 1e-9))
        return int(np.argmax(self.hit_rate + bonus))
    def update(self, action, reward):
        self.counts[action] += 1.0
        self.hits[action] += float(np.isclose(reward, REWARDS["hit"]))
        self.hit_rate[action] = self.hits[action] / self.counts[action]


class ThompsonAgent:
    def __init__(self, n_bands, seed = 0):
        self.n_bands = n_bands
        self.reset(seed)
    def reset(self, seed):
        self.successes = np.ones(self.n_bands)
        self.failures = np.ones(self.n_bands)
        self.rng = np.random.default_rng(seed)
    def decide(self, obs, info):
        p = self.rng.beta(self.successes, self.failures)
        return int(np.argmax(p))
    def update(self, action, reward):
        if np.isclose(reward, REWARDS["hit"]):
            self.successes[action] += 1.0
        else:
            self.failures[action] += 1.0


def main():

    st.set_page_config(page_title = "EW PPO vs Baselines", layout = "wide")

    st.title("Spectrum Sensing: PPO vs Heuristics")
    st.caption("Same EWenv seeds per episode for every agent - apples to apples.")

    n_episodes = st.sidebar.slider("Episodes per agent", 1, 20, 3)
    seed0 = st.sidebar.slider("Base seed", 0, 999, 0)
    run = st.sidebar.button("Run comparison")

    agents = {
        "PPO (actor.pt)": PPO_Frozen(),
        "Energy-greedy": EnergyAgent(10),
        "UCB1": UCB1Agent(10),
        "Thompson": ThompsonAgent(10),
        "Random": RandomAgent(10),
    }

    if run:
        env = EWenv()
        all_rows = {}
        for name, agent in agents.items():
            with st.spinner(f"Evaluating {name}..."):
                all_rows[name] = run_agent(agent, env, n_episodes, seed0, None)

        st.subheader("Per-agent summary")
        table = []
        for name, rows in all_rows.items():
            table.append({
                "agent": name,
                "mean reward": float(np.mean([r["reward"] for r in rows])),
                "detections": int(np.sum([r["detections"] for r in rows])),
                "false alarms": int(np.sum([r["false_alarms"] for r in rows])),
                "misses": int(np.sum([r["misses"] for r in rows])),
            })
        st.dataframe(table)

        st.subheader("Episode reward by agent")
        fig, ax = plt.subplots()
        for name, rows in all_rows.items():
            xs = [r["reward"] for r in rows]
            ax.plot(range(1, len(xs) + 1), xs, marker = "o", label = name)
        ax.set_xlabel("episode")
        ax.set_ylabel("reward")
        ax.xaxis.set_major_locator(MaxNLocator(integer = True))
        ax.legend()
        st.pyplot(fig)

        st.subheader("Detections by agent")
        fig2, ax2 = plt.subplots()
        for name, rows in all_rows.items():
            xs = [r["detections"] for r in rows]
            ax2.plot(range(1, len(xs) + 1), xs, marker = "o", label = name)
        ax2.set_xlabel("episode")
        ax2.set_ylabel("detections")
        ax2.xaxis.set_major_locator(MaxNLocator(integer = True))
        ax2.legend()
        st.pyplot(fig2)


if __name__ == "__main__":
    main()
