#!/usr/bin/env python3

import numpy as np

from drlnd.core.common.ring_buffer import ContiguousRingBuffer


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

        # NOTE(yycho0108): "done" is technically boolean, but mapping here to uint8
        # To support torch conversion (bool is not supported on certain versions.)
        self.dtype = np.dtype([
            ('state', np.float32, state_size),
            ('action', np.int32),
            ('reward', np.float32),
            ('next_state', np.float32, state_size),
            ('done', np.uint8)])
        # self.memory = {name : ContiguousRingBuffer(capacity=buffer_size, dtype=self.dtype.fields[name][0]) for name in  self.dtype.names}
        self.memory = ContiguousRingBuffer(
            capacity=buffer_size, dtype=self.dtype)
        self.batch_size = batch_size

        # Manipulate random engine.
        self.rng = np.random.RandomState(seed)
        self.nadd = 0
        self.nquery = 0

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        entry = np.array((state, action, reward, next_state, done),
                         dtype=self.dtype)
        self.memory.append(entry)
        self.nadd += 1

    def extend(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        entry = np.empty(len(state), dtype=self.dtype)
        entry['state'] = state
        entry['action'] = action
        entry['reward'] = reward
        entry['next_state'] = next_state
        entry['done'] = done
        self.memory.extend(entry)
        self.nadd += len(state)

    def sample(self, indices=None):
        """Randomly sample a batch of experiences from memory."""
        if indices is None:
            indices = self.rng.randint(len(self.memory), size=self.batch_size)

        # NOTE(yycho0108): It is much more favorable to index by name first here,
        # to prevent creation of multiple copies since the output must ultimately be contiguous.
        # Since indexing by the field name will merely create a view, applying the selection indices last
        # Will create the final contiguous copy without the intermediate memory.
        # print(self.memory['state'].base is self.memory.data_) # True
        # print(self.memory[indices].base is self.memory.data_) # False
        # print(self.memory['state'][indices].base is self.memory.data_) # False

        out = [(self.memory[name][indices]) for name in self.dtype.fields]
        self.nquery += 1
        return out, indices

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
