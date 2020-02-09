#!/usr/bin/env python3

import numpy as np
import os
import time
import gym
import json
import sys
from tqdm import tqdm

from drlnd.core.agents.base_agent import AgentBase
from drlnd.core.common.ring_buffer import ContiguousRingBuffer
from drlnd.core.common.logger import get_default_logger
from drlnd.core.common.epsilon import ExponentialEpsilon, LinearEpsilon

logger = get_default_logger()


class TestSettings(dict):
    def __init__(self, **kwargs):
        self.num_episodes = 16
        self.directory = '/tmp/'
        self.render = True
        self.fps = 100.0
        self.enabled = True
        self.__dict__.update(kwargs)
        dict.__init__(self, self.__dict__)

    def __str__(self):
        return self.__dict__.__str__()

    def __repr__(self):
        return self.__dict__.__repr__()


def test(env: gym.Env, agent: AgentBase, settings: TestSettings):
    # Initialize variables for logging.
    agent.load(settings.directory)
    scores = ContiguousRingBuffer(capacity=128)
    for i_episode in tqdm(range(settings.num_episodes)):
        # Initialize episode
        state = env.reset()
        total_reward = 0

        # Interact with the environment until done.
        done = False
        step = 0
        while not done:
            action = agent.select_action(state)
            if settings.render:
                env.render()
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
            time.sleep(1.0 / settings.fps)
            logger.debug('{}:{}'.format(step, action))
            step += 1

    return scores
