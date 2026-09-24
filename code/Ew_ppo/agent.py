import torch
from torch import nn

#Action

class Actor(nn.Module):
    def __init__(self, obs_size, action_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )
    def forward(self, x):
        x = self.flatten(x)
        # Per-sample standardization: raw observations are uncalibrated
        # energy magnitudes that vary wildly in scale between archives.
        # Normalize each sample to zero mean / unit std so the fully
        # connected body learns scale-invariant features. Fixed transform
        # (no learnable params, no buffers), so it behaves identically at
        # train and inference and old state_dicts remain loadable.
        x = (x - x.mean(dim = -1, keepdim = True)) / (
            x.std(dim = -1, keepdim = True) + 1e-6
        )
        logits = self.linear_relu_stack(x)

        return logits


# Critic

class Critic(nn.Module):
    def __init__(self, obs_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256,1)
        )
    def forward(self, x):
        x = self.flatten(x)
        x = (x - x.mean(dim = -1, keepdim = True)) / (
            x.std(dim = -1, keepdim = True) + 1e-6
        )
        value = self.linear_relu_stack(x)

        return value
