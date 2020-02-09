#!/usr/bin/env python3

import time
import torch as tr
import numpy as np
from drlnd.core.common.replay_buffer import ReplayBuffer
from baselines.deepq.replay_buffer import ReplayBuffer as GymReplayBuffer


def main():
    state_size = 17
    action_size = 4
    buffer_size = 1024
    batch_size = 32
    num_steps = 4096
    num_samples = 1024
    num_repeat = 10

    gym_memory = GymReplayBuffer(buffer_size)
    memory = ReplayBuffer(state_size, action_size, buffer_size, batch_size, 0)

    # Make some convenient aliases.
    n = num_steps
    ns = state_size
    na = action_size

    # Generate random experiences ...
    states = np.zeros((n, ns), dtype=np.float32)
    actions = np.random.randint(0, na, n)
    rewards = np.random.uniform(0, 1, n)
    next_states = np.zeros((n, ns), dtype=np.float32)
    dones = np.random.randint(2, size=n, dtype=np.bool)

    ts=[]
    ts.append(time.time())

    print('Memory')
    for _ in range(num_repeat):
        for s0, a, r, s1, d in zip(states, actions, rewards, next_states, dones):
            memory.add(s0, a, r, s1, d)
    ts.append(time.time())
    for _ in range(num_repeat):
        for _ in range(num_samples):
            sample = memory.sample()
    ts.append(time.time())

    print('Gym-Memory')
    for _ in range(num_repeat):
        for s0, a, r, s1, d in zip(states, actions, rewards, next_states, dones):
            gym_memory.add(s0, a, r, s1, d)
    ts.append(time.time())
    for _ in range(num_repeat):
        for _ in range(num_samples):
            sample = gym_memory.sample(batch_size)
    ts.append(time.time())

    print('Result')
    print(np.diff(ts))

if __name__ == '__main__':
    main()
