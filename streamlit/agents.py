import os

import numpy as np
import torch
from torch.distributions import Categorical


class RandomAgent:

    def __init__(self, n_bands, seed = 0):
        self.n_bands = n_bands
        self.reset(seed)

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)

    def decide(self, observation, step_info):
        return int(self.rng.integers(0, self.n_bands))


class EnergyBaseline:

    # Deterministic greedy over smoothed band energy. Fully reproducible
    # (no RNG), and a strong "operator heuristic" stable baseline.

    def __init__(self, n_bands, seed = 0):
        self.n_bands = n_bands

    def reset(self, seed):
        pass

    def decide(self, observation, step_info):
        return int(np.argmax(observation[0]))


class UCB1Agent:

    def __init__(self, n_bands, seed = 0):
        self.n_bands = n_bands
        self.reset(seed)

    def reset(self, seed):
        self.counts = np.zeros(self.n_bands, dtype = np.float64)
        self.hits = np.zeros(self.n_bands, dtype = np.float64)
        self.hit_rate = np.zeros(self.n_bands, dtype = np.float64)
        self.t = 0

    def decide(self, observation, step_info):
        self.t += 1
        if self.t <= self.n_bands:
            return int((self.t - 1) % self.n_bands)
        log_t = np.log(float(self.t))
        bonus = np.sqrt(2.0 * log_t / np.maximum(self.counts, 1e-9))
        return int(np.argmax(self.hit_rate + bonus))


class ThompsonAgent:

    def __init__(self, n_bands, seed = 0):
        self.n_bands = n_bands
        self.reset(seed)

    def reset(self, seed):
        self.successes = np.ones(self.n_bands, dtype = np.float64)
        self.failures = np.ones(self.n_bands, dtype = np.float64)

    def decide(self, observation, step_info):
        draws = np.random.default_rng(step_info["step"]).beta(
            self.successes,
            self.failures,
        )
        return int(np.argmax(draws))


class PPOAgent:

    def __init__(self, n_bands, seed = 0, device = "cpu"):
        self.n_bands = n_bands
        self.device = device
        self.reset(seed)

    def reset(self, seed):
        pass

    def decide(self, observation, step_info):
        obs_t = torch.as_tensor(
            observation,
            dtype = torch.float32,
            device = self.device,
        ).reshape(1, -1)
        with torch.no_grad():
            logits = self.actor(obs_t)
            return int(Categorical(logits = logits).sample().item())

    def load_actor(self, actor_path, obs_size = 80, action_size = 10):
        from agent import Actor

        self.actor = Actor(obs_size, action_size).to(self.device)

        self.actor.load_state_dict(
            torch.load(
                actor_path,
                map_location = self.device,
                weights_only = True,
            )
        )
        self.actor.eval()
