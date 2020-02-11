#!/usr/bin/env python3

import os
from pathlib import Path
import numpy as np
import random

import hydra
import torch
import torch.nn.functional as F
import torch.optim as optim

from drlnd.core.common.logger import get_default_logger
from drlnd.core.common.replay_buffer import ReplayBuffer
from drlnd.core.agents.base_agent import AgentBase
from drlnd.core.common.util import count_boundaries

logger = get_default_logger()


class DQNAgentSettings(dict):
    def __init__(self, **kwargs):
        self.buffer_size = int(5e4)  # replay buffer size
        self.batch_size = 32        # minibatch size
        self.gamma = 1.0           # discount factor
        self.learning_rate = 5e-4   # learning rate
        self.update_period = 1    # how often to update the network
        # for soft update of target parameters
        self.train_delay_step = 10
        self.target_update_factor = 0.01
        self.target_update_period = 8
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.network = {'class': 'drlnd.core.networks.mlp.QNetworkMLP'}
        self.seed = None
        self.__dict__.update(kwargs)
        dict.__init__(self, self.__dict__)

    def __str__(self):
        return self.__dict__.__str__()

    def __repr__(self):
        return self.__dict__.__repr__()


class DQNAgent(AgentBase):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, **kwargs):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.settings = DQNAgentSettings(**kwargs)
        self.seed = self.settings.seed
        random.seed(self.seed)

        # Q-Network
        logger.debug('Settings : {}'.format(self.settings))
        logger.debug('Loading Network as : {}'.format(self.settings.network))
        self.qnetwork_local = hydra.utils.instantiate(self.settings.network, state_size, action_size).to(
            self.settings.device)
        self.qnetwork_target = hydra.utils.instantiate(self.settings.network, state_size, action_size).to(
            self.settings.device)
        self.optimizer = optim.Adam(
            self.qnetwork_local.parameters(), lr=self.settings.learning_rate)

        # Replay memory
        self.memory = ReplayBuffer(
            state_size, action_size, int(self.settings.buffer_size), self.settings.batch_size, self.settings.seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory.
        # TODO(yycho0108): validate batch_dim equal for all inputs.
        batch_dim = state.shape[:-len(self.state_size)]
        if len(batch_dim) == 0:
            self.memory.add(state, action, reward, next_state, done)
        else:
            self.memory.extend(state, action, reward, next_state, done)
        step_size = 1 if len(batch_dim) == 0 else len(state)
        prev_step = self.t_step
        self.t_step += step_size

        # Skip learning if insufficient memory size.
        should_learn = (len(self.memory) >= self.settings.batch_size and
                        self.t_step > self.settings.train_delay_step)

        # If learning is not enabled, just apply the step size here and return.
        if not should_learn:
            return

        # Learn every `update_period` time steps.
        # NOTE(yycho0108): for parallalized runs, number of added steps
        # may not equal to 1. In such a case, the number of updates
        # should be determined based on the step size.
        num_updates = count_boundaries(
            prev_step, step_size, self.settings.update_period)
        for _ in range(num_updates):
            experiences = self.memory.sample()
            experiences = [torch.from_numpy(e).to(
                self.settings.device) for e in experiences]
            self.learn(experiences, self.settings.gamma)

        # Update target network similarly every `target_update_period` time steps.
        # Here, similar logic is applied to count the number of target updates.
        # Note that here the update factor is pre-calculated
        # as an exponential and only applied once.
        num_target_updates = count_boundaries(
            prev_step, step_size, self.settings.target_update_period)
        if num_target_updates > 0:
            target_update_factor = (1.0 -
                                    (1.0 - self.settings.target_update_factor) ** num_target_updates)
            self.soft_update(self.qnetwork_local,
                             self.qnetwork_target, target_update_factor)

    def select_action(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        batch_dim = state.shape[:-len(self.state_size)]
        state = torch.from_numpy(state).float().to(self.settings.device)
        if len(batch_dim) == 0:
            state = state.unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        p = np.random.uniform(0.0, 1.0, size=batch_dim)
        action = np.where(p > eps,
                          np.argmax(action_values.cpu().data.numpy(), axis=-1),
                          np.random.choice(self.action_size, size=batch_dim))
        if len(batch_dim) == 0:
            action = np.asscalar(np.squeeze(action, 0))
        return action

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            Q_target_next = self.qnetwork_target(
                next_states).detach().max(1)[0].unsqueeze(1)
            rewards.unsqueeze_(1)
            dones.unsqueeze_(1)
            Q_target = rewards + (1 - dones.float()) * (gamma * Q_target_next)

        Q_local = self.qnetwork_local(states).gather(
            1, actions.long().unsqueeze(1))

        loss = F.mse_loss(Q_local, Q_target)
        # loss = F.smooth_l1_loss(Q_local, Q_target)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data)

    @staticmethod
    def _get_checkpoint(path, i=None, search=False):
        # Try with default first.
        if i is None:
            filename = '{}/checkpoint.pth'.format(path)
        else:
            filename = '{}/checkpoint-{}.pth'.format(path, i)

        # Optionally, fallback to search directory for any available checkpoint file.
        if not Path(filename).exists() and search:
            files = Path(path).glob('**/*.pth')
            filename = sorted(files, key=os.path.getmtime)[-1]
            logger.info('Fallback to loading from : {}'.format(filename))

        return filename

    def load(self, path='', i=None):
        filename = self._get_checkpoint(path, i, search=True)
        state_dict = torch.load(filename)
        logger.info('Agent loading from : {}'.format(filename))
        self.qnetwork_local.load_state_dict(state_dict)

    def save(self, path='', i=None):
        filename = self._get_checkpoint(path, i, search=False)
        logger.info('Agent saving to : {}'.format(filename))
        torch.save(self.qnetwork_local.state_dict(), filename)
