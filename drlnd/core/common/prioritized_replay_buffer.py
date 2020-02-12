#!/usr/bin/env python3

import numpy as np
import numba as nb

from drlnd.core.common.ring_buffer import ContiguousRingBuffer


@nb.njit()
def _resample_wheel(weights: np.ndarray, size: int, max_weight: float, out: np.ndarray):
    n = len(weights)
    weights = np.asarray(weights)
    index = np.random.randint(n)

    beta = 0.0
    if max_weight is None:
        max_weight = weights.max()

    # Assuming n >> size,
    # take ~large steps to de-correlate output data.
    step = int(np.ceil(n / size))

    for i in range(size):
        beta += np.random.random() * 2.0 * max_weight
        while beta > weights[index]:
            beta -= weights[index]
            index = (index + step) % n
        out[i] = index


def resample_wheel(weights, size=1, max_weight=None, out=None):
    if out is None:
        out = np.empty(size, np.int32)
    if max_weight is None:
        max_weight = weights.max()
    _resample_wheel(weights, size, max_weight, out)
    return out


class PrioritizedReplayBuffer(object):
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
            ('done', np.uint8),
            ('priority', np.float32),
        ])
        self.memory = ContiguousRingBuffer(
            capacity=buffer_size, dtype=self.dtype)
        self.batch_size = batch_size
        self.max_priority = 1.0
        self.fields = ['state', 'action', 'reward', 'next_state', 'done']

        # Manipulate random engine.
        self.rng = np.random.RandomState(seed)
        self.nadd = 0
        self.nquery = 0

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        entry = np.array((state, action, reward, next_state, done, self.max_priority),
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
        entry['priority'] = np.full(len(state), self.max_priority)
        self.memory.extend(entry)
        self.nadd += len(state)

    def sample(self, indices=None):
        """Randomly sample a batch of experiences from memory."""
        if indices is None:
            indices = resample_wheel(self.memory['priority'], self.batch_size)
        out = [(self.memory[name][indices]) for name in self.fields]
        self.nquery += 1
        return out, indices

    def update_priorities(self, indices, priorities):
        """ Update priority """
        self.max_priority = max(self.max_priority, priorities.max())
        self.memory['priority'][indices] = priorities

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


def main():
    from matplotlib import pyplot as plt
    weights = np.random.uniform(size=128)
    indices = resample_wheel(weights, 12)
    print(indices)
    plt.plot(weights / weights.sum(), '*-')
    plt.hist(indices, bins=128, density=True)
    plt.show()


if __name__ == '__main__':
    main()
