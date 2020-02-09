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


class TrainSettings(dict):
    def __init__(self, **kwargs):
        now = time.strftime("%Y%m%d-%H%M%S")

        self.num_episodes = 1000
        self.log_period = 100
        self.directory = '/tmp/train-{}'.format(now)
        self.__dict__.update(kwargs)
        dict.__init__(self, self.__dict__)

    def save(self, directory=None):
        if directory is None:
            directory = self.directory
        os.makedirs(directory, exist_ok=True)
        filename = os.path.join(directory, 'settings.json')
        logger.debug('Saving to : {}'.format(filename))
        with open(filename, 'w') as f:
            json.dump({k: str(v) for k, v in self.__dict__.items()}, f)

    def __str__(self):
        return self.__dict__.__str__()

    def __repr__(self):
        return self.__dict__.__repr__()


def train(env: gym.Env, agent: AgentBase, settings: TrainSettings):
    # Initialize variables for logging.
    scores = ContiguousRingBuffer(capacity=128)
    max_avg_score = -np.inf

    # eps = LinearEpsilon(0.8 * settings.num_episodes)
    eps = ExponentialEpsilon(0.99, 0.05, 0.8 * settings.num_episodes, True)

    t = tqdm(range(settings.num_episodes))
    for i_episode in t:
        # Initialize episode
        state = env.reset()
        total_reward = 0

        # Interact with the environment until done.
        done = False
        while not done:
            action = agent.select_action(state, eps(i_episode))
            next_state, reward, done, _ = env.step(action)
            # NOTE(yycho0108): agent.step() traines the agent
            # FIXME(yycho0108): rename?
            agent.step(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state

        # Save the final score.
        scores.append(total_reward)

        t.set_postfix(score=np.mean(scores.array))
        # Optionally enable printing episode stats.
        if i_episode % settings.log_period == 0:
            # Compute statistics.
            avg_score = np.mean(scores.array)
            if avg_score > max_avg_score:
                max_avg_score = avg_score

            # Print statistics.
            logger.info("Episode {}/{} | Max Avg: {:.2f} | Eps : {:.2f}".format(
                i_episode, settings.num_episodes, max_avg_score, eps(i_episode)))
            # sys.stdout.flush()

    # Save results.
    os.makedirs(settings.directory, exist_ok=True)
    agent.save(settings.directory)
    settings.save(settings.directory)

    return scores
