#!/usr/bin/env python3
"""2-shuffle_data module
contains the function shuffle_data
"""
import numpy as np


def shuffle_data(X, Y):
    """Shuffles the data points in two matrices the same way.
    Args:
        X: `numpy.ndarray` of shape (m, nx), first array to be shuffled.
            m: `int`, the number of data points.
            nx: `int`, the number of feature columns in our data.
        Y: `numpy.ndarray` of shape (m, ny), second array to be shuffled.
            m: `int`, the same number of data points as in X.
            ny: `int`, the number of feature columns in Y.

    Returns:
        X_shuffled, Y_shuffled: `tuple`, Shuffled X and Y matrices.
    """
    # numpy.random.permutation also return a permuted range
    shufled_rows = np.random.permutation(X.shape[0])
    return X[shufled_rows], Y[shufled_rows]
