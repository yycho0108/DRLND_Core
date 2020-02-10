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
    # cfg = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    logger = get_root_logger(level=logging.DEBUG)
    logger.info('Config : {}'.format(cfg))
    logger.info('Agent : {}'.format(cfg.agent))

    env = gym.make(cfg.env)
    # cfg.agent.params.network = cfg.network
    agent = hydra.utils.instantiate(
        cfg.agent, env.observation_space.shape, env.action_space.n)

    if cfg.train.enabled:
        train_settings = TrainSettings(**cfg.train)
        # train_settings.directory = os.getcwd()
        logger.info(train_settings)
        # train(env, agent, train_settings)

    if cfg.test.enabled:
        test_settings = TestSettings(**cfg.test)
        # test_settings.directory = os.getcwd()
        logger.info(test_settings)
        # test(env, agent, test_settings)

    env.close()


if __name__ == '__main__':
    main()
