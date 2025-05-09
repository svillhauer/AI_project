# rl_loop_agent.py

import torch
import torch.nn as nn

class RLLoopAgent(nn.Module):
    def __init__(self, obs_size, action_size):
        super(RLLoopAgent, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
