#!/usr/bin/env python3

import importlib
import numpy as np


def lerp(a, b, w):
    """ linear interpolation """
    return a + w*(b-a)


def import_class(class_string: str):
    """ Import class from string description """
    module_name, class_name = class_string.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
