#!/usr/bin/env python3

import os
import gym
import hydra
import logging

from drlnd.core.pipeline.test import (TestSettings, test)
from drlnd.core.pipeline.train import (TrainSettings, train)
from drlnd.core.common.logger import get_root_logger
from drlnd.core.common.path import get_project_root, get_config_file
from drlnd.core.common.util import import_class


@hydra.main(config_path=get_config_file())
def main(cfg) -> None:
    logger = get_root_logger(level=logging.INFO)

    env = gym.make(cfg.env)
    agent = hydra.utils.instantiate(
        cfg.agent, env.observation_space.shape, env.action_space.n)

    if cfg.train.enabled:
        train_settings = TrainSettings(**cfg.train)
        logger.info(train_settings)
        if train_settings.num_env > 1:
            def env_fn(index): return gym.make(cfg.env)
            train(env_fn, agent, train_settings)
        else:
            train(env, agent, train_settings)

    if cfg.test.enabled:
        test_settings = TestSettings(**cfg.test)
        logger.info(test_settings)
        agent.load(test_settings.directory)
        scores = test(env, agent, test_settings)
        logger.info('Test scores : {}'.format(scores))

    env.close()


if __name__ == '__main__':
    main()
