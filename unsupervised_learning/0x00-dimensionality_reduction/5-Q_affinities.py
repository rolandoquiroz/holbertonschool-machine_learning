#!/usr/bin/env python3
"""
module 5-Q_affinities
contains function P_affinities
"""
import numpy as np


def Q_affinities(Y):
    """
    Calculates the Q affinities
    """
    D = (np.sum(Y ** 2, axis=1) - 2 * np.matmul(Y, Y.T) +
         np.sum(Y ** 2, axis=1)[:, np.newaxis])
    num = 1/(1 + D)
    np.fill_diagonal(num, 0.)
    Q = num / np.sum(num)
    return Q, num
