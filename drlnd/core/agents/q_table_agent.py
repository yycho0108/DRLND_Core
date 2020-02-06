#!/usr/bin/env python3

from abc import ABC, abstractmethod

import numpy as np
from enum import Enum
from collections import defaultdict
import pickle

from drlnd.core.agents.base_agent import AgentBase
from drlnd.core.common.epsilon import EpsilonBase, IncrementalEpsilon
from drlnd.core.common.util import lerp
from drlnd.core.common.table import DictQTable, TiledQTable


class QControlMethod(Enum):
    kMethodSarsa0 = 0
    kMethodSarsaMax = 1
    kMethodSarsaExpect = 2


class QTableAgent(AgentBase):

    def __init__(self, num_actions: int,
                 epsilon: EpsilonBase,
                 control: QControlMethod = QControlMethod.kMethodSarsaExpect,
                 alpha: float = 0.1,
                 gamma: float = 1.0,
                 Q: DictQTable = None
                 ):
        self.num_actions_ = num_actions
        if isinstance(epsilon, IncrementalEpsilon):
            self.eps_ = epsilon
        else:
            self.eps_ = IncrementalEpsilon(epsilon)
        self.ctrl_ = control
        self.alpha_ = alpha
        self.gamma_ = gamma

        # NOTE(yycho0108): instead of initializing to zero,
        # Consider alternative values to enable "optimistic"
        # Q-table.

        if Q is None:
            Q = DictQTable(self.num_actions_)
        self.Q_ = Q

    def get_action_probs(self, state):
        n = self.num_actions_
        probs = np.empty(n)
        if self.Q_.has(state):
            # E-greedy probability
            eps = self.eps_()
            best_action = self.Q_[state].argmax()
            probs.fill(eps / len(probs))  # -> sum to eps
            probs[best_action] += (1.0 - eps)
        else:
            # Uniform probability for unknown states
            probs.fill(1.0/n)
        return probs

    def select_action(self, state):
        """
        Select action based on the current state and the estimated q-table.
        """
        # E-greedy policy
        if np.random.random() < self.eps_():
            return np.random.choice(self.num_actions_)
        return np.argmax(self.Q_.get(state))

    def step(self, state, action, reward, next_state, done):
        # Update q-table
        q_old = self.Q_.get(state, action)

        # Determine expected q value based on the
        # Control method settings.
        q_next = 0.0
        if done:
            # FIXME(yycho0108): somewhat fragile method
            # To track table training progress.
            self.eps_.increment_index()
        else:
            if self.ctrl_ == QControlMethod.kMethodSarsa0:
                # NOTE(yycho0108): does not currently work
                q_next = self.Q_.get(next_state, next_action)
            elif self.ctrl_ == QControlMethod.kMethodSarsaMax:
                q_next = np.max(self.Q_.get(next_state))
            elif self.ctrl_ == QControlMethod.kMethodSarsaExpect:
                probs = self.get_action_probs(next_state)
                q_next = self.Q_.get(next_state).dot(probs)

        q_new = reward + self.gamma_ * q_next
        self.Q_.update(state, action, q_new, self.alpha_)

    def save(self, filename='q.pkl'):
        data = (self.num_actions_, self.eps_, self.ctrl_,
                self.alpha_, self.gamma_, self.Q_)
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filename='q.pkl'):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.num_actions_, self.eps_, self.ctrl_, self.alpha_, self.gamma_, self.Q_ = data

    @classmethod
    def from_file(cls, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        num_actions, epsilon, control, alpha, gamma, Q = data
        return QTableAgent(num_actions, epsilon, control, alpha, gamma, Q)
