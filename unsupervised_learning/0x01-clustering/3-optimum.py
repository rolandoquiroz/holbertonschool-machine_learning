#!/usr/bin/env python3
"""
module 3-optimum
contains function optimum
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Tests for the optimum number of clusters by variance
    """
    if type(X) is not np.ndarray or len(X.shape) is not 2:
        return None, None

    if kmax is None:
        kmax = X.shape[0]

    if type(kmin) is not int or not 1 < kmin < X.shape[0] - 1:
        return None, None

    if type(kmax) is not int or not 1 < kmax <= X.shape[0]:
        return None, None

    if kmin >= kmax:
        return None, None

    if type(iterations) is not int or iterations < 1:
        return None, None

    results = []
    d_vars = []
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))
        if k == kmin:
            var_min = variance(X, C)
        d_vars.insert(len(d_vars), var_min - variance(X, C))
    return results, d_vars
