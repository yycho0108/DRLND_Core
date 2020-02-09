#!/usr/bin/env python3

from abc import ABC, abstractmethod


class EpsilonBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, i):
        raise NotImplementedError('')


class InverseEpsilon(ABC):
    def __init__(self):
        super().__init__()

    def __call__(self, i):
        return (1.0 / i)


class ExponentialEpsilon(object):
    def __init__(self, eps0, eps1, n, clip=False):
        super().__init__()
        self.eps0_ = eps0
        self.eps1_ = eps1
        self.n_ = n
        self.clip_ = clip

        self.decay_ = (eps1 / eps0) ** (1.0 / n)

    def __call__(self, i):
        if self.clip_ and i >= self.n_:
            return 0.0
        return self.eps0_ * (self.decay_ ** i)


class LinearEpsilon(object):
    def __init__(self, n):
        super().__init__()
        self.n_ = n

    def __call__(self, i):
        return 1.0 - (i / float(self.n_))


class ConstantEpsilon(object):
    def __init__(self, value):
        super().__init__()
        self.value_ = value

    def __call__(self, _):
        return self.value_


class IncrementalEpsilon(EpsilonBase):
    def __init__(self, eps):
        self.eps_ = eps
        self.index_ = 0

    def increment_index(self):
        self.index_ += 1

    def __call__(self):
        return self.eps_(self.index_)
