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
    X = np.concatenate((X, Y), axis=1)
    X = np.random.permutation(X)
    X_shuffled = X[:, :Y.shape[1]]
    Y_shuffled = X[:, Y.shape[1]:]
    return X_shuffled, Y_shuffled
