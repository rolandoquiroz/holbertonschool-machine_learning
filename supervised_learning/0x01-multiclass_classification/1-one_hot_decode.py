#!/usr/bin/env python3
"""1-one_hot_decode"""

import numpy as np


def one_hot_decode(one_hot):
    """Converts a one-hot matrix into a vector of labels
    Args:
        one_hot (numpy.ndarray): One-hot encoded array with shape (classes, m)
        where classes is the maximum number of classes
        and m is the number of examples
    Returns:
        (numpy.ndarray): Array with shape (m, ) containing the numeric labels
        for each example, or None on failure
    """
    if (type(one_hot) is not np.ndarray or
            len(one_hot) < 1 or
            all(type(i) in [int, float] for i in one_hot)):
        return None
    decoded = np.argmax(one_hot, axis=0)
    return decoded
