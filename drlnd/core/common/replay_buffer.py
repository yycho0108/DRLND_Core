#!/usr/bin/env python3

import torch
import numpy as np
from .ring_buffer import ContiguousRingBuffer


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, state_size, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            state_size (int or tuple): dimension of state
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size

        self.dtype = np.dtype([
            ('state', np.float32, state_size),
            ('action', np.int32),
            ('reward', np.float32),
            ('next_state', np.float32, state_size),
            ('done', np.bool)])

        self.memory = ContiguousRingBuffer(
            capacity=buffer_size, dtype=self.dtype)
        self.batch_size = batch_size
        self.seed = np.random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = np.array((state, action, reward, next_state, done),
                     dtype=self.dtype)
        self.memory.append(e)

    def sample(self, device=torch.cuda.current_device()):
        """Randomly sample a batch of experiences from memory."""
        indices = np.random.randint(len(self.memory), size=self.batch_size)
        experiences = self.memory.array[indices]

        def _get(name):
            return torch.from_numpy(np.ascontiguousarray(experiences[name]))

        states = _get('state').float().to(device)
        actions = _get('action').long().to(device)
        rewards = _get('reward').float().to(device)
        next_states = _get('next_state').float().to(device)
        dones = _get('done').float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


def main():
    memory = ReplayBuffer(17, 4, 512, 32, 0)

    n = 2
    ns = 17
    na = 4

    states = np.zeros((n, ns), dtype=np.float32)
    actions = np.random.randint(0, na, n)
    rewards = np.random.uniform(0, 1, n)
    next_states = np.zeros((n, ns), dtype=np.float32)
    dones = np.random.randint(2, size=n, dtype=np.bool)
    for s0, a, r, s1, d in zip(states, actions, rewards, next_states, dones):
        memory.add(s0, a, r, s1, d)
    sample = (memory.sample())


if __name__ == '__main__':
    main()
