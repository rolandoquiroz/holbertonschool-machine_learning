#!/usr/bin/env python3
"""0-norm_constants module
contains the function normalization_constants
"""
import numpy as np


def normalization_constants(X):
    """Calculates the normalization (standardization) constants of a matrix.
    Args:
        X: `numpy.ndarray` of shape (m, nx) to normalize.
            m: `int`, the number of data points.
            nx: `int`, the number of feature columns in our data.

    Returns:
        me, s: `tuple`, mean and standard deviation of each feature,
            respectively.
    """
    me = np.mean(X, axis=0)
    s = np.std(X, axis=0)
    return me, s
