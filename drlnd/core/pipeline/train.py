#!/usr/bin/env python3

import numpy as np
import os
import time
import gym
import json
import sys
import functools
from tqdm import tqdm, tnrange

from drlnd.core.common.util import is_notebook, count_boundaries
from drlnd.core.agents.base_agent import AgentBase
from drlnd.core.common.ring_buffer import ContiguousRingBuffer
from drlnd.core.common.prioritized_replay_buffer import PrioritizedReplayBuffer
from drlnd.core.common.logger import get_default_logger
from drlnd.core.common.epsilon import ExponentialEpsilon, LinearEpsilon

from baselines.common.vec_env import SubprocVecEnv

logger = get_default_logger()


class TrainSettings(dict):
    def __init__(self, **kwargs):
        now = time.strftime("%Y%m%d-%H%M%S")
        self.num_episodes = 1000
        self.log_period = 100
        self.save_period = 100
        self.directory = '/tmp/train-{}'.format(now)
        self.enabled = True
        self.load = ''
        self.num_env = 0
        self.__dict__.update(kwargs)
        dict.__init__(self, self.__dict__)

    def __str__(self):
        return self.__dict__.__str__()

    def __repr__(self):
        return self.__dict__.__repr__()


def train_multi(env: gym.Env, agent: AgentBase, settings: TrainSettings):
    # Initialize variables for logging.
    scores = ContiguousRingBuffer(capacity=128)
    max_avg_score = -np.inf

    # Ensure settings.directory exists for logging / saving.
    os.makedirs(settings.directory, exist_ok=True)
    # Optionally load from existing checkpoint.
    if settings.load:
        agent.load(settings.load)

    # Instantiate vectorized environment.
    if isinstance(env, SubprocVecEnv):
        # No further action is required.
        pass
    elif isinstance(env, gym.Env):
        # Cannot
        logger.error("Unable to broadcast single environment {}".format(env))
    else:
        # Assume that env is a constructor function.
        env = SubprocVecEnv([functools.partial(env, i)
                             for i in range(settings.num_env)])

    # Initialize handlers for data collection.
    total_rewards = np.zeros(settings.num_env, dtype=np.float32)
    dones = np.zeros(settings.num_env, dtype=np.uint8)
    states = env.reset()
    # FIXME(yycho0108): EPS should be configurable.
    # eps = LinearEpsilon(0.8 * settings.num_episodes)
    eps = ExponentialEpsilon(0.99, 0.05, 0.8 * settings.num_episodes, True)

    i_episode = 0
    pbar = tqdm(total=settings.num_episodes)
    while i_episode < settings.num_episodes:
        # Reset the environments that are done, so that
        # At each moment the agent is always dealing with a live-state.
        # SubprocVecEnv.reset() does not allow granular control.
        for s, d, e in zip(states, dones, env.remotes):
            if not d:
                continue
            e.send(('reset', None))
            # FIXME(yycho0108): Applying a reshape here as e.recv()
            # Was seen to return a list for whatever reason.
            # May silently allow an error to pass through.
            s[:] = np.reshape(e.recv(), s.shape)
        scores.extend(total_rewards[dones == True])
        total_rewards[dones == True] = 0.0
        num_done = dones.sum()
        dones[:] = False

        # Process each state and interact with each env.
        actions = agent.select_action(states, eps(i_episode))
        next_states, rewards, dones, _ = env.step(actions)
        agent.step(states, actions, rewards, next_states, dones)
        total_rewards += rewards
        states = next_states

        # Increment episode counts accordingly.
        pbar.set_postfix(score=np.mean(scores.array))

        # Optionally enable printing episode statistics.
        # The logging happens at each crossing of the discretized log-period boundary.
        if count_boundaries(i_episode, num_done, settings.log_period) > 0:
            # Compute statistilcs.
            avg_score = np.mean(scores.array)
            if avg_score > max_avg_score:
                max_avg_score = avg_score

            # Print statistics.
            logger.info("Episode {}/{} | Max Avg: {:.2f} | Eps : {:.2f}".format(
                i_episode, settings.num_episodes, max_avg_score, eps(i_episode)))
            if isinstance(agent.memory,  PrioritizedReplayBuffer):
                logger.info('mp : {} vs {}'.format(
                    agent.memory.max_priority, agent.memory.memory.array['priority'].max()))

        # Save agent checkpoint as well.
        if count_boundaries(i_episode, num_done, settings.save_period) > 0:
            agent.save(settings.directory, i_episode + num_done)

        i_episode += num_done
        pbar.update(num_done)
    pbar.close()

    # Save results and return.
    agent.save(settings.directory)
    return scores


def train_single(env: gym.Env, agent: AgentBase, settings: TrainSettings):
    # Initialize variables for logging.
    scores = ContiguousRingBuffer(capacity=128)
    max_avg_score = -np.inf

    # Ensure settings.directory exists for logging / saving.
    os.makedirs(settings.directory, exist_ok=True)
    # Optionally load from existing checkpoint.
    if settings.load:
        agent.load(settings.load)

    # Instantiate vectorized environment.
    if isinstance(env, gym.Env):
        # No further action is required.
        pass
    else:
        # Assume that env is a constructor function.
        env = env()

    # FIXME(yycho0108): EPS should be configurable.
    # eps = LinearEpsilon(0.8 * settings.num_episodes)
    eps = ExponentialEpsilon(0.99, 0.05, 0.8 * settings.num_episodes, True)

    if is_notebook():
        t = tnrange(settings.num_episodes)
    else:
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
            # NOTE(yycho0108): agent.step() trains the agent.
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

        if i_episode % settings.save_period == 0:
            agent.save(settings.directory, i_episode)

    # Save results and return.
    agent.save(settings.directory)
    return scores


def train(env: gym.Env, agent: AgentBase, settings: TrainSettings):
    if settings.num_env > 1:
        return train_multi(env, agent, settings)
    else:
        return train_single(env, agent, settings)
