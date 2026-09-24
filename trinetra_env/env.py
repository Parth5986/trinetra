from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import EnvConfig
from tsrd_loader import TsrdLoader
from emitter import TsrdEmitters

from typing import Optional

class EWenv(gym.Env):

    metadata = {"render_modes": ["human"],"render_fps": 10}

    def __init__(self,config: Optional[EnvConfig] = None,render_mode: Optional[str] = None):
        super().__init__()
        self.config = config or EnvConfig()
        self.render_mode = render_mode

        self.loader = TsrdLoader(self.config.tsrd_path)

        self.all_transmitter = self.loader.get_all()
        self.transmitter_by_mode = self.loader.frequency_mode

        self.in_range_transmitters = [
            cfg for cfg in self.all_transmitter
            if self._frequency_in_range(cfg)
        ]

        if not self.in_range_transmitters:
            raise ValueError(
                f"No transmitter within receiver range "
                f"[{self.config.min_frequency}, {self.config.max_frequency}] MHz"
            )

        self.band_edges = np.linspace(self.config.min_frequency,self.config.max_frequency,self.config.n_bands + 1)

        self.band_centers = (self.band_edges[:-1] + self.band_edges[1:]) / 2.0

        # Action space

        self.action_space = spaces.Discrete(self.config.n_bands)

        # Observation space

        self.obs_low = np.zeros((self.config.n_bands,8),dtype = "f")
        self.obs_high = np.ones((self.config.n_bands,8),dtype = "f")

        self.obs_high[:,2] = self.config.max_steps # hit count
        self.obs_high[:,3] = self.config.max_steps # miss count

        self.observation_space = spaces.Box(low = self.obs_low,high = self.obs_high,dtype = "f")

        # Env state

        self.emitters = []

        self.current_step = 0
        self.time_s = 0.0

        self.energy_history = np.zeros(self.config.n_bands,dtype = "f")

        self.staleness = np.ones(self.config.n_bands,dtype = "f")

        self.hit = np.zeros(self.config.n_bands,dtype = "f")

        self.misses = np.zeros(self.config.n_bands,dtype = "f")

        self.detection_history = np.zeros(self.config.n_bands,dtype = "f")

        self.occupancy = np.zeros(self.config.n_bands,dtype = "f")

        self.last_band_state = np.zeros(self.config.n_bands,dtype = bool)

    def reset(self,*,seed = None,option = None,):

        super().reset(seed = seed)

        self.current_step = 0
        self.time_s = 0.0

        self.energy_history[:] = 0.0
        self.staleness[:] = 1.0
        self.hit[:] = 0.0
        self.misses[:] = 0.0
        self.detection_history[:] = 0.0
        self.occupancy[:] = 0.0

        self.selected_config = self.select_emitters()

        self.emitters = [
            TsrdEmitters(config, self.np_random)
            for config in self.selected_config
        ]

        observation = self.get_observation()

        info = {
            "n_emitters": len(self.emitters),
            "emitter_modes": [emitter.freq_mode
                              for emitter in self.emitters],
        }

        return observation,info

    def _frequency_in_range(self, config) -> bool:

        freqs = np.asarray(config["frequency_config"]["freqs_mhz"],dtype = "f")

        lo = float(freqs.min())
        hi = float(freqs.max())

        return hi >= self.config.min_frequency and lo <= self.config.max_frequency

    def select_emitters(self):
        n = self.config.n_emitters

        if self.config.balanced_sampling:
            modes = list(self.transmitter_by_mode.keys())

            selected = []

            modes = list(self.np_random.permutation(modes))

            for mode in modes:
                if len(selected) >=n:
                    break

                candidates = [
                    cfg for cfg in self.transmitter_by_mode[mode]
                    if self._frequency_in_range(cfg)
                ]

                if not candidates:
                    continue

                index = self.np_random.integers(len(candidates))

                selected.append(candidates[index])

            while len(selected) < n:
                index = self.np_random.integers(len(self.in_range_transmitters))
                selected.append(self.in_range_transmitters[index])

            return selected

        # Non-balanced sampling: pick uniformly from all in-range transmitters.
        selected = list(self.np_random.choice(
            self.in_range_transmitters,
            size = min(n, len(self.in_range_transmitters)),
            replace = False,
        ))

        return selected

    def frequency_to_band(self,frequency_mhz: float):

        if frequency_mhz<self.config.min_frequency or frequency_mhz>self.config.max_frequency:
            return None

        band = np.searchsorted(self.band_edges,frequency_mhz,side="right")-1

        band = int(np.clip(band,0,self.config.n_bands - 1))

        return band

    def get_band_activity(self):

        band_energy = np.zeros(self.config.n_bands,dtype = "f")

        band_occupied = np.zeros(self.config.n_bands,dtype = bool)

        band_power = np.full(self.config.n_bands,-150.0,dtype = "f")

        for emitter in self.emitters:
            frequencies = emitter.get_active_frequencies()

            power_dbm = emitter.get_signal_power_dbm(self.time_s)

            for frequency in frequencies:

                band = self.frequency_to_band(frequency)

                if band is None:
                    continue

                band_occupied[band] = True

                band_power[band] = max(band_power[band],power_dbm)

        for band in range(self.config.n_bands):
            if band_occupied[band]:
                normalized = (band_power[band] - self.config.sensitive_dbm)/100.0

                band_energy[band] = np.clip(normalized,0.0,1.0)

        return (band_energy,band_power,band_occupied)

    def get_observation(self):

        energy, power, occupied = self.get_band_activity()

        decay = self.config.history_decay

        self.energy_history = (decay * self.energy_history + (1.0 - decay) * energy)

        threat = np.clip(0.6 * self.energy_history + 0.4 * self.occupancy,0.0,1.0)

        snr = np.clip(power - self.config.sensitive_dbm,0.0,100.0)/100.0

        observation = np.column_stack(
            (
            energy,
            np.clip(self.staleness/self.config.max_steps,0.0,1.0),
            self.hit,
            self.misses,
            threat,
            snr,
            self.detection_history,
            self.occupancy)
        ).astype(np.float32)

        return observation

    def step(self, action):

        action = int(action)

        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action}")

        self.current_step +=1

        energy, power, occupied = self.get_band_activity()

        reward = 0.0

        selected_occupied = occupied[action]

        if selected_occupied:

            # Detection probability based on energy

            detection_probability = np.clip(
                energy[action],
                0.05,
                0.99,
            )

            detected = (
                    self.np_random.random()
                    < detection_probability
            )

            if detected:

                reward = (
                    self.config.hit
                )

                self.hit[action] += 1

                self.detection_history[action] = (
                        0.9
                        * self.detection_history[action]
                        + 0.1
                )

            else:

                reward = (
                    self.config.miss
                )

                self.misses[action] += 1

        else:

            # Empty band

            false_alarm_probability = 0.02

            false_alarm = (
                    self.np_random.random()
                    < false_alarm_probability
            )

            if false_alarm:

                reward = (
                    self.config.false_alarm
                )

                self.misses[action] += 1

            else:

                reward = (
                    self.config.empty_band
                )

        self.staleness += 1.0

        self.staleness[action] = 0.0


        # occupancy is a belief, updated only from the agent's own scan measurement (no ground truth)

        self.occupancy *= 0.95

        self.occupancy[action] += 0.3 * float(energy[action])

        np.minimum(self.occupancy, 1.0, out = self.occupancy)


        self.detection_history *= 0.98


        for emitter in self.emitters:
            emitter.step(
                self.config.step_time_sec
            )

        self.time_s += (
            self.config.step_time_sec
        )


        terminated = False

        truncated = (
                self.current_step
                >= self.config.max_steps
        )

        observation = (
            self.get_observation()
        )

        info = {

            "step": self.current_step,

            "selected_band": action,

            "band_occupied": bool(
                selected_occupied
            ),

            "reward": float(reward),

            "active_emitters": len(
                self.emitters
            ),

            "emitter_modes": [
                emitter.freq_mode
                for emitter in self.emitters
            ],
        }

        if self.render_mode == "human":
            self.render()

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )


    def render(self):

        energy, power, occupied = self.get_band_activity()

        print("\n" + "=" * 60)

        print(
            f"STEP: {self.current_step}"
        )

        print(
            f"TIME: {self.time_s:.2f} seconds"
        )

        print("-" * 60)

        for i in range(
                self.config.n_bands
        ):
            low = self.band_edges[i]
            high = self.band_edges[i + 1]

            current = energy[i]
            history = self.energy_history[i]
            belief = self.occupancy[i]

            print(
                f"Band {i:02d} | "
                f"{low:8.1f}–{high:8.1f} MHz | "
                f"Live: {current:.3f} | "
                f"dBm: {power[i]:7.1f} | "
                f"Hist: {history:.3f} | "
                f"Belief: {belief:.3f}"
            )

        print("=" * 60)

    def close(self):

        pass
















