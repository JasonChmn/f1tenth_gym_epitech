"""Q-network for DQN: MLP mapping observations to Q-values per action."""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """2-layer MLP Q-network. No CNN.
    
    Input: observation vector
    Output: Q-value for each action
    """

    def __init__(self, obs_dim: int, action_dim: int, hid_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, action_dim),
        )

    def forward(self, x):
        # obs → Q-values per action
        return self.net(x)