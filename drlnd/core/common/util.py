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


def is_notebook():
    """
    Check if insite a notebook.
    Source:
    https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook/39662359#39662359
    """
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
