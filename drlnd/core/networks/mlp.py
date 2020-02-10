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


class QNetworkMLP(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden=[], seed=0, name='mlp'):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden (list(int)): Dimensions of intermediate layers.
            seed (int): Random seed for weight initialization.
        """
        super().__init__()
        self.name = name

        if seed is not None:
            self.seed = torch.manual_seed(seed)

        dimensions = [np.prod(state_size)] + list(hidden)  # + [action_size]
        self.net = nn.ModuleList()
        for lhs, rhs in zip(dimensions[:-1], dimensions[1:]):
            self.net.append(LinearRelu(lhs, rhs))
        # Add final layer for classification.
        self.net.append(nn.Linear(hidden[-1], action_size))

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        for i, l in enumerate(self.net):
            x = self.net[i](x)
        return x
