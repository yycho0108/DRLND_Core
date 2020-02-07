#!/usr/bin/env python3
import numpy as np


def lerp(a, b, w):
    """ linear interpolation """
    return a + w*(b-a)
