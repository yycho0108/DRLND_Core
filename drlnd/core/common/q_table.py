#!/usr/bin/env python3

from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np

from drlnd.core.common.util import lerp
from drlnd.core.common.logger import get_default_logger

logger = get_default_logger()

def create_tiling_grid(low, high, bins=(10, 10), offsets=(0.0, 0.0)):
    """Define a uniformly-spaced grid that can be used for tile-coding a space.

    Parameters
    ----------
    low : array_like
        Lower bounds for each dimension of the continuous space.
    high : array_like
        Upper bounds for each dimension of the continuous space.
    bins : tuple
        Number of bins or tiles along each corresponding dimension.
    offsets : tuple
        Split points for each dimension should be offset by these values.

    Returns
    -------
    grid : list of array_like
        A list of arrays containing split points for each dimension.
    """
    # TODO: Implement this
    grid = [(np.linspace(l, h, b+1)[1:-1]+o)
            for (l, h, b, o) in zip(low, high, bins, offsets)]
    return grid


def create_tilings(low, high, tiling_specs):
    """Define multiple tilings using the provided specifications.

    Parameters
    ----------
    low : array_like
        Lower bounds for each dimension of the continuous space.
    high : array_like
        Upper bounds for each dimension of the continuous space.
    tiling_specs : list of tuples
        A sequence of (bins, offsets) to be passed to create_tiling_grid().

    Returns
    -------
    tilings : list
        A list of tilings (grids), each produced by create_tiling_grid().
    """
    # TODO: Implement this
    return [create_tiling_grid(low, high, bins, offsets) for (bins, offsets) in tiling_specs]


def discretize(sample, grid):
    """Discretize a sample as per given grid.

    Parameters
    ----------
    sample : array_like
        A single sample from the (original) continuous space.
    grid : list of array_like
        A list of arrays containing split points for each dimension.

    Returns
    -------
    discretized_sample : array_like
        A sequence of integers with the same number of dimensions as sample.
    """
    return [np.searchsorted(g, s, side='right') for s, g in zip(sample, grid)]


def tile_encode(sample, tilings, flatten=False):
    """Encode given sample using tile-coding.

    Parameters
    ----------
    sample : array_like
        A single sample from the (original) continuous space.
    tilings : list
        A list of tilings (grids), each produced by create_tiling_grid().
    flatten : bool
        If true, flatten the resulting binary arrays into a single long vector.

    Returns
    -------
    encoded_sample : list or array_like
        A list of binary vectors, one for each tiling, or flattened into one.
    """
    # TODO: Implement this
    out = [discretize(sample, grid) for grid in tilings]
    if flatten:
        out = np.ravel(out)
    return out


class QTable(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def has(self, state, action=None):
        raise NotImplementedError('{},{}'.format(state, action))

    @abstractmethod
    def get(self, state, action=None):
        raise NotImplementedError('{},{}'.format(state, action))

    @abstractmethod
    def set(self, state, action, value):
        raise NotImplementedError('{},{},{}'.format(state, action, value))


class DictQTable(QTable):
    def __init__(self, action_size):
        self.action_size_ = action_size
        self.Q_ = defaultdict(lambda: np.zeros(action_size))

    def has(self, state, action=None):
        return self.Q_.__contains__(state)

    def get(self, state, action=None):
        if action is None:
            return self.Q_[state]
        return self.Q_[state][action]

    def set(self, state, action, value):
        self.Q_[state][action] = value

    def merge(self, table):
        if isinstance(table, DictQTable):
            self.Q_.update(table.Q_)
        else:
            self.Q_.update(table)

    def update(self, state, action, value, alpha=None):
        if alpha is not None:
            value = lerp(self.Q_[state][action], value, alpha)
        self.set(state, action, value)


class NDArrayQTable(QTable):

    """Simple Q-table."""

    def __init__(self, state_size: tuple, action_size: int, track=False):
        """Initialize Q-table.

        Parameters
        ----------
        state_size : tuple
            Number of discrete values along each dimension of state space.
        action_size : int
            Number of discrete actions in action space.
        """
        if np.isscalar(state_size):
            state_size = (state_size,)
        self.state_size = state_size
        self.action_size = action_size

        # TODO: Create Q-table, initialize all Q-values to zero
        # Note: If state_size = (9, 9), action_size = 2, q_table.shape should be (9, 9, 2)
        shape = state_size + (action_size,)
        self.Q_ = np.zeros(shape, dtype=np.float32)

        if track:
            self.N_ = np.zeros(shape, dtype=np.int32)
        else:
            self.N_ = None

    @staticmethod
    def _key(state, action):
        if np.isscalar(state):
            key = (state, action)
        else:
            key = tuple(state) + (action,)
        return tuple(key)

    def has(self, state, action=None):
        key = self._key(state, action)
        # Pythonic check - to deal with negative indices, etc.
        try:
            self.Q_[key]
            return True
        except IndexError:
            return False

    def get(self, state, action=None):
        if action is None:
            return self.Q_[tuple(state)]
        key = self._key(state, action)
        out = self.Q_[key]
        return self.Q_[key]

    def set(self, state, action, value):
        key = self._key(state, action)
        if self.N_ is not None:
            self.N_[key] += 1
        self.Q_[key] = value
        # return self.Q_.__setitem__(key, value)

    def merge(self, table):
        if isinstance(table, NDArrayQTable):
            # Somewhat principled average over number of accesses.
            if self.N_ is not None and table.N_ is not None:
                self.Q_ = np.average([self.Q_, table.Q_], [self.N_, table.N_])
        else:
            # Simple average.
            self.Q_ = lerp(self.Q_, table.Q_, 0.5)

    def update(self, state, action, value, alpha=None):
        if alpha is not None:
            key = self._key(state, action)
            value = lerp(self.Q_[key], value, alpha)
        self.set(state, action, value)


class TiledQTable(QTable):
    """Composite Q-table with an internal tile coding scheme."""

    def __init__(self, low, high, tiling_specs, action_size):
        """Create tilings and initialize internal Q-table(s).

        Parameters
        ----------
        low : array_like
            Lower bounds for each dimension of state space.
        high : array_like
            Upper bounds for each dimension of state space.
        tiling_specs : list of tuples
            A sequence of (bins, offsets) to be passed to create_tilings() along with low, high.
        action_size : int
            Number of discrete actions in action space.
        """
        self.tilings = create_tilings(low, high, tiling_specs)
        self.state_sizes = [tuple(len(splits)+1 for splits in tiling_grid)
                            for tiling_grid in self.tilings]
        self.action_size = action_size
        self.q_tables = [NDArrayQTable(state_size, action_size)
                         for state_size in self.state_sizes]
        logger.debug("TiledQTable(): no. of internal tables = {}".format(len(self.q_tables)))

    def has(self, state, action=None):
        return np.all([q_table.has(state, action) for q_table in self.q_tables])

    def get(self, state, action=None):
        """Get Q-value for given <state, action> pair.

        Parameters
        ----------
        state : array_like
            Vector representing the state in the original continuous space.
        action : int
            Index of desired action.

        Returns
        -------
        value : float
            Q-value of given <state, action> pair, averaged from all internal Q-tables.
        """
        # TODO: Encode state to get tile indices
        # TODO: Retrieve q-value for each tiling, and return their average
        d_states = tile_encode(state, self.tilings)
        q_values = [q_table.get(d_state, action)
                    for (q_table, d_state) in zip(self.q_tables, d_states)]
        return np.mean(q_values, axis=0)

    def set(self, state, action, value):
        d_states = tile_encode(state, self.tilings)
        for q_table, d_state in zip(self.q_tables, d_states):
            q_table.set(d_state, action, value)

    def update(self, state, action, value, alpha=0.1):
        """Soft-update Q-value for given <state, action> pair to value.

        Instead of overwriting Q(state, action) with value, perform soft-update:
            Q(state, action) = alpha * value + (1.0 - alpha) * Q(state, action)

        Parameters
        ----------
        state : array_like
            Vector representing the state in the original continuous space.
        action : int
            Index of desired action.
        value : float
            Desired Q-value for <state, action> pair.
        alpha : float
            Update factor to perform soft-update, in [0.0, 1.0] range.
        """
        # TODO: Encode state to get tile indices
        # TODO: Update q-value for each tiling by update factor alpha
        d_states = tile_encode(state, self.tilings)
        for q_table, d_state in zip(self.q_tables, d_states):
            q_table.update(d_state, action, value, alpha)
