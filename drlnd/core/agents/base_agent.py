#!/usr/bin/env python3

from abc import ABC, abstractmethod


class AgentBase(ABC):
    def __init__(self):
        """ Initialize agent.

        Params
        ======
        """
        pass

    @abstractmethod
    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        raise NotImplementedError('Cannot call abstract method select_action')

    @abstractmethod
    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        raise NotImplementedError('Cannot call abstract method step')
