import numpy as np
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from drlnd.core.common.replay_buffer import ReplayBuffer
from drlnd.core.networks.simple_q_network import QNetwork
from drlnd.core.agents.base_agent import AgentBase


class DQNAgentSettings(dict):
    def __init__(self, **kwargs):
        self.buffer_size = int(5e4)  # replay buffer size
        self.batch_size = 32        # minibatch size
        self.gamma = 1.0           # discount factor
        self.learning_rate = 5e-4   # learning rate
        self.update_period = 1    # how often to update the network
        # for soft update of target parameters
        self.train_delay_step = 1000
        self.target_update_factor = 0.01
        self.target_update_period = 500
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.network = QNetwork
        self.__dict__.update(kwargs)
        dict.__init__(self, self.__dict__)

    def __str__(self):
        return self.__dict__.__str__()

    def __repr__(self):
        return self.__dict__.__repr__()


class DQNAgent(AgentBase):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, settings=DQNAgentSettings(), seed=0):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.settings = settings

        # Q-Network
        self.qnetwork_local = settings.network(
            state_size, action_size, seed).to(self.settings.device)
        self.qnetwork_target = settings.network(
            state_size, action_size, seed).to(self.settings.device)
        self.optimizer = optim.Adam(
            self.qnetwork_local.parameters(), lr=self.settings.learning_rate)

        # Replay memory
        self.memory = ReplayBuffer(
            state_size, action_size, settings.buffer_size, settings.batch_size, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.settings.update_period
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.settings.batch_size:
                experiences = self.memory.sample()
                experiences = [torch.from_numpy(e).to(
                    self.settings.device) for e in experiences]
                self.learn(experiences, self.settings.gamma)

    def select_action(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(
            0).to(self.settings.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

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

        # loss = F.mse_loss(Q_local, Q_target)
        loss = F.smooth_l1_loss(Q_local, Q_target)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        if self.t_step > self.settings.train_delay_step and self.t_step % self.settings.target_update_period == 0:
            self.soft_update(self.qnetwork_local,
                             self.qnetwork_target, self.settings.target_update_factor)

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

    def load(self, path=''):
        filename = '{}/checkpoint.pth'.format(path)
        state_dict = torch.load(filename)
        self.qnetwork_local.load_state_dict(state_dict)

    def save(self, path=''):
        filename = '{}/checkpoint.pth'.format(path)
        torch.save(self.qnetwork_local.state_dict(), filename)
