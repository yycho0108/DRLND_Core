#!/usr/bin/env python3

from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearRelu(nn.Module):
    def __init__(self, D_in, D_out):
        super().__init__()

        # Ensure 1D input/output.
        #D_in = np.prod(D_in)
        #D_out = np.prod(D_out)

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
        if seed is not None:
            self.seed = torch.manual_seed(seed)

        self.net = nn.Sequential(OrderedDict([
            ('fc1', LinearRelu(np.prod(state_size), 256)),
            ('fc2', nn.Linear(256, action_size))
        ]))

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.net(state)
