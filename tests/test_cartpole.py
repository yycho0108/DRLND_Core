#!/usr/bin/env python3

import argparse
import gym
import logging

from drlnd.core.agents.q_table_agent import QTableAgent
from drlnd.core.train.pipeline import pipeline, TrainSettings
from drlnd.core.common.epsilon import ExponentialEpsilon, IncrementalEpsilon
from drlnd.core.common.q_table import TiledQTable
from drlnd.core.common.logger import get_root_logger, get_default_logger


def main():
    logger = get_root_logger(logging.DEBUG)

    settings = TrainSettings()
    settings.num_episodes = 10000
    settings.log_period = 100
    # env = gym.make('CartPole-v1')
    env = gym.make('Acrobot-v1')

    eps = ExponentialEpsilon(0.99, 0.05, 0.8 * settings.num_episodes, True)

    # Configure tiles
    n_bins = 9
    bins = tuple([n_bins]*env.observation_space.shape[0])
    offset_pos = (env.observation_space.high - env.observation_space.low)/(3*n_bins)

    tiling_specs = [(bins, -offset_pos),
                    (bins, tuple([0.0]*env.observation_space.shape[0])),
                    (bins, offset_pos)]

    table = TiledQTable(env.observation_space.low,
                        env.observation_space.high, tiling_specs, env.action_space.n)

    agent = QTableAgent(env.action_space.n, eps, Q=table, alpha=0.02, gamma=0.99)
    pipeline(env, agent, settings)


if __name__ == '__main__':
    main()
