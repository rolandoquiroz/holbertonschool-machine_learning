#!/usr/bin/env python3
"""
Module 1-correlation
contains the correlation function"""

import numpy as np


def correlation(C):
    """Calculates a correlation matrix
    Args:
        C (numpy.ndarray): with shape (d, d) containing a covariance matrix
            d is the number of dimensions
    Returns
        A numpy.ndarray of shape (d, d) containing the correlation matrix
    """
    if type(C) is not np.ndarray:
        raise TypeError('C must be a numpy.ndarray')

    if len(C.shape) is not 2 or C.shape[0] != C.shape[1]:
        raise ValueError('C must be a 2D square matrix')

    variance = np.diag(1 / np.sqrt(np.diag(C)))
    corr = np.matmul(np.matmul(variance, C), variance)

    return corr
