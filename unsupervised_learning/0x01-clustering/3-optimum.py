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
    if type(iterations) is not int or iterations < 1:
        return None, None
    if type(kmin) is not int or kmin < 1 or kmin >= X.shape[0]:
        return None, None
    if type(kmax) is not int or kmax < 1 or kmax > X.shape[0]:
        return None, None
    if kmax is not None and kmin >= kmax:
        return None, None

    last = X.shape[0] if kmax is None else kmax + 1
    results = []
    d_vars = []
    for k in range(kmin, last):
        C, clss = kmeans(X, k, iterations)
        var = variance(X, C)
        if kmin == k:
            var_min = variance(X, C)
        results.append((C, clss))
        d_vars.append(var_min - var)
    return results, d_vars
