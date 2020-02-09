#!/usr/bin/env python3

import time
import argparse
import gym
import logging

from drlnd.core.agents.q_table_agent import QTableAgent
from drlnd.core.agents.dqn_agent import DQNAgent, DQNAgentSettings
from drlnd.core.train.pipeline import pipeline, TrainSettings
from drlnd.core.common.epsilon import ExponentialEpsilon, IncrementalEpsilon, LinearEpsilon
from drlnd.core.common.q_table import TiledQTable
from drlnd.core.common.logger import get_root_logger, get_default_logger


def main():
    logger = get_root_logger(logging.DEBUG)

    settings = TrainSettings(agent='dqn', agent_settings=DQNAgentSettings())
    settings.num_episodes = 100000
    settings.log_period = 100

    print('Settings : {}'.format(settings))
    # env = gym.make('CartPole-v1')
    env = gym.make('Acrobot-v1')

    # eps = ExponentialEpsilon(0.99, 0.05, 0.8 * settings.num_episodes, True)
    eps = LinearEpsilon(0.8 * settings.num_episodes)

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

        agent = QTableAgent(env.action_space.n, eps,
                            Q=table, alpha=0.02, gamma=0.99)
    elif settings.agent == 'dqn':
        agent = DQNAgent(env.observation_space.shape, env.action_space.n)

    if settings.train:
        pipeline(env, agent, settings)

    if True:
        # agent.load('/tmp/train-20200208-220512/')
        for i_episode in (range(16)):
            # Initialize episode
            state = env.reset()
            total_reward = 0

            # Interact with the environment until done.
            done = False
            while not done:
                action = agent.select_action(state)
                env.render()
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                state = next_state
                time.sleep(0.1)
    return


if __name__ == '__main__':
    main()
