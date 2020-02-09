#!/usr/bin/env python3

import time
import argparse
import gym
import logging

from drlnd.core.agents.q_table_agent import QTableAgent
from drlnd.core.agents.dqn_agent import DQNAgent, DQNAgentSettings
from drlnd.core.pipeline.train import train, TrainSettings
from drlnd.core.pipeline.test import test, TestSettings
from drlnd.core.common.epsilon import ExponentialEpsilon, IncrementalEpsilon, LinearEpsilon
from drlnd.core.common.q_table import TiledQTable
from drlnd.core.common.logger import get_root_logger, get_default_logger


class AppSettings(object):
    def __init__(self, args={}, train_args={}, test_args={}):
        self.train = True
        self.test = False
        self.train_settings = TrainSettings(**train_args)
        self.test_settings = TestSettings(**test_args)
        self.__dict__.update(args)

    def __str__(self):
        return self.__dict__.__str__()

    def __repr__(self):
        return self.__dict__.__repr__()


def main():
    logger = get_root_logger(logging.DEBUG)

    settings = AppSettings(
        dict(agent='dqn'),
        dict(num_episodes=100000, log_period=100, agent='dqn'),
        dict(num_episodes=4, agent='dqn', directory='/tmp/train-20200208-222140/')
    )
    settings.train = False
    settings.test = True

    print('Settings : {}'.format(settings))
    env = gym.make('CartPole-v1')
    # env = gym.make('Acrobot-v1')

    # Configure tiles
    if settings.agent == 'q_table':
        n_bins = 5
        bins = tuple([n_bins]*env.observation_space.shape[0])
        offset_pos = (env.observation_space.high -
                      env.observation_space.low)/(3*n_bins)

        tiling_specs = [(bins, -offset_pos),
                        (bins, tuple([0.0]*env.observation_space.shape[0])),
                        (bins, offset_pos)]

        table = TiledQTable(env.observation_space.low,
                            env.observation_space.high, tiling_specs, env.action_space.n)

        # FIXME(yycho0108): q_table eps is not supported right now.
        agent = QTableAgent(env.action_space.n, eps,
                            Q=table, alpha=0.02, gamma=0.99)
    elif settings.agent == 'dqn':
        agent = DQNAgent(env.observation_space.shape, env.action_space.n)

    if settings.train:
        train(env, agent, settings.train_settings)

    if settings.test:
        agent.load(settings.test_settings.directory)
        # agent.load('/tmp/train-20200208-220512/')
        test(env, agent, settings.test_settings)

    env.close()
    return


if __name__ == '__main__':
    main()
