#!/usr/bin/env python3

import gym
import hydra
import logging

from drlnd.core.pipeline.test import (TestSettings, test)
from drlnd.core.pipeline.train import (TrainSettings, train)
from drlnd.core.common.logger import get_root_logger
from drlnd.core.common.path import get_project_root, get_config_file


@hydra.main(config_path=get_config_file())
def main(cfg) -> None:
    logger = get_root_logger(level=logging.INFO)

    env = gym.make(cfg.env)
    agent = hydra.utils.instantiate(
        cfg.agent, env.observation_space.shape, env.action_space.n)

    if cfg.train.enabled:
        train_settings = TrainSettings(**cfg.train)
        logger.info('Train Settings : {}'.format(train_settings))
        train(env, agent, train_settings)

    if cfg.test.enabled:
        test_settings = TestSettings(**cfg.test)
        logger.info('Test Settings : {}'.format(test_settings))
        test(env, agent, test_settings)
    env.close()


if __name__ == '__main__':
    main()
