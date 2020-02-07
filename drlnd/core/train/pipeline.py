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

logger = get_default_logger()


class TrainSettings(object):
    def __init__(self, **kwargs):
        now = time.strftime("%Y%m%d-%H%M%S")

        self.num_episodes = 1000
        self.log_period = 100
        self.directory = '/tmp/train-{}'.format(now)
        self.train = True

    def save(self, directory=None):
        if directory is None:
            directory = self.directory
        os.makedirs(directory, exist_ok=True)
        filename = os.path.join(directory, 'settings.json')
        logger.debug('Saving to : {}'.format(filename))
        with open(filename, 'w') as f:
            json.dump(self.__dict__, f)


def pipeline(env: gym.Env, agent: AgentBase, settings: TrainSettings):
    # Initialize variables for logging.
    scores = ContiguousRingBuffer(capacity=128)
    max_avg_score = -np.inf

    for i_episode in tqdm(range(settings.num_episodes)):
        # Initialize episode
        state = env.reset()
        total_reward = 0

        # Interact with the environment until done.
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            if settings.train:
                agent.step(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state

        # Save the final score.
        scores.append(total_reward)

        # Optionally enable printing episode stats.
        if settings.train:
            if i_episode % settings.log_period == 0:
                # Compute statistics.
                avg_score = np.mean(scores.array)
                if avg_score > max_avg_score:
                    max_avg_score = avg_score

                # Print statistics.
                logger.info("Episode {}/{} | Max Average Score: {}".format(
                    i_episode, settings.num_episodes, max_avg_score))
                # sys.stdout.flush()

    # Save results.
    os.makedirs(settings.directory, exist_ok=True)
    agent.save(settings.directory)
    settings.save(settings.directory)

    return scores
