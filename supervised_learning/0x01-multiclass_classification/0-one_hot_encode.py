#!/usr/bin/env python3
"""0-one_hot_encode module"""

import numpy as np


def one_hot_encode(Y, classes):
    """Converts a numeric label vector into a one-hot matrix
    Args:
        Y (numpy.ndarray): Numeric class labels array with shape (m,)
            where m is the number of examples
        classes (int): Maximum number of classes found in Y
    Returns:
        One-hot encoding of Y with shape (classes, m), or None on failure
    """
    if type(Y) is not numpy.ndarray or len(Y) == 0:
        return None
    if type(classes) is not int or classes <= np.amax(Y):
        return None
    one_hot = np.eye(classes)[Y].T
    return (one_hot)
