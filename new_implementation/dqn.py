import torch.nn as nn

import config

# --------------
# Deep Q-Network
# --------------
# - uses a MLP as architecture
# - takes state observation and return actions values as a Q-function approximation
class DQNetwork(nn.Module):
    def __init__(self):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features=config.STATE_DIM, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=config.ACTION_DIM)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        o = self.relu(self.fc1(x))
        o = self.relu(self.fc2(o))
        o = self.fc3(o)
        return o