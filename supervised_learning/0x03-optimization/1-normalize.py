#!/usr/bin/env python3
"""1-normalize module
contains the function normalize
"""


def normalize(X, m, s):
    """Normalizes (standardizes) a matrix.
    Args:
        X: `numpy.ndarray` of shape (d, nx) to normalize.
            d: `int`, the number of data points.
            nx: `int`, the number of feature columns in our data.
        m: `numpy.ndarray` of shape (nx,) that contains the mean of all
            features of X.
        s: `numpy.ndarray` of shape (nx,) that contains the standard deviation
            of all features of X

    Returns:
        X_normalized: `numpy.ndarray`, normalized X matrix.
    """
    X_normalized = (X - m)/s
    return X_normalized
