#!/usr/bin/env python3

import numpy as np
from enum import Enum
from collections import defaultdict

from agents.base import AgentBase
from common.epsilon import EpsilonBase
from common.util import lerp

import pickle


class QControlMethod(Enum):
    METHOD_SARSA_0 = 0
    METHOD_SARSA_MAX = 1
    METHOD_SARSA_EXPECT = 2


class IncrementalEpsilon(EpsilonBase):
    def __init__(self, eps):
        self.eps_ = eps
        self.index_ = 0

    def increment_index(self):
        self.index_ += 1

    def __call__(self):
        return self.eps_(self.index_)


class QTableAgent(AgentBase):

    def __init__(self, num_actions: int,
                 epsilon: EpsilonBase,
                 control: QControlMethod = QControlMethod.METHOD_SARSA_EXPECT,
                 alpha: float = 0.1,
                 gamma: float = 1.0,
                 Q=None
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
        self.Q_ = defaultdict(lambda: np.zeros(self.num_actions_))
        if Q is not None:
            self.Q_.update(Q)

    def get_action_probs(self, state):
        n = self.num_actions_
        probs = np.empty(n)
        if state in self.Q_:
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
        return self.Q_[state].argmax()

    def step(self, state, action, reward, next_state, done):
        # Update q-table
        q_old = self.Q_[state][action]

        # Determine expected q value based on the
        # Control method settings.
        q_next = 0.0
        if done:
            # FIXME(yycho0108): somewhat fragile method
            # To track table training progress.
            self.eps_.increment_index()
        else:
            if self.ctrl_ == QControlMethod.METHOD_SARSA_0:
                q_next = self.Q_[next_state][next_action]
            elif self.ctrl_ == QControlMethod.METHOD_SARSA_MAX:
                q_next = self.Q_[next_state].max()
            elif self.ctrl_ == QControlMethod.METHOD_SARSA_EXPECT:
                probs = self.get_action_probs(next_state)
                q_next = self.Q_[next_state].dot(probs)

        q_new = reward + self.gamma_ * q_next
        self.Q_[state][action] = lerp(q_old, q_new, self.alpha_)

    def save(self, filename='q.pkl'):
        data = (self.num_actions_, self.eps_, self.ctrl_,
                self.alpha_, self.gamma_, dict(self.Q_))
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
