import numpy as np
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from core.common.replay_buffer import ReplayBuffer
from core.networks.simple_q_network import QNetwork
from .base_agent import AgentBase


class DQNAgentSettings(object):
    def __init__(self):
        self.buffer_size = int(1e5)  # replay buffer size
        self.batch_size = 64         # minibatch size
        self.gamma = 0.99            # discount factor
        self.tau = 1e-3              # for soft update of target parameters
        self.learning_rate = 5e-4               # learning rate
        self.update_period = 4        # how often to update the network
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.network = QNetwork


class DQNAgent(AgentBase):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, settings=DQNAgentSettings(), seed):
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
            if len(self.memory) > settings.batch_size:
                experiences = self.memory.sample(device=self.settings.device)
                self.learn(experiences, self.settings.gamma)

    def act(self, state, eps=0.):
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

        # TODO: compute and minimize the loss
        with torch.no_grad():
            Q_target_next = self.qnetwork_target(
                next_states).detach().max(1)[0].unsqueeze(1)
            Q_target = rewards + (1 - dones) * (gamma * Q_target_next)
        Q_local = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_local, Q_target)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local,
                         self.qnetwork_target, self.settings.tau)

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
