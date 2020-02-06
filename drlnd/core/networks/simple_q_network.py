from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearRelu(nn.Module):
    def __init__(self, D_in, D_out):
        super().__init__()
        self.linear = torch.nn.Linear(D_in, D_out)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.net = nn.Sequential(OrderedDict([
            ('fc1', LinearRelu(state_size, 64)),
            ('fc2', LinearRelu(64, 32)),
            ('fc2', LinearRelu(32, 16)),
            ('fc3', nn.Linear(16, action_size))
        ]))

        "*** YOUR CODE HERE ***"

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.net(state)
