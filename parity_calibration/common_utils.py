"""
Common utilities for the parity calibration package.
"""

import numpy as np


def logit(arr):
    threshold = 1e-5
    arr = np.minimum(np.maximum(arr, threshold), 1 - threshold)
    return np.log(arr / (1 - arr))


def logistic_transform(x):
    threshold = -100
    x = np.clip(x, a_min=threshold, a_max=None)
    return 1 / (1 + np.exp(-x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
