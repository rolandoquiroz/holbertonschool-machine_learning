#!/usr/bin/env python3
"""
module 3-optimum
contains function optimum
"""
import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Tests for the optimum number of clusters by variance"""
    try:
        if type(X) is not np.ndarray or len(X.shape) is not 2:
            return None, None
        if type(iterations) is not int or iterations < 1:
            return None, None
        if kmax is None:
            kmax = X.shape[0]
        if type(kmin) is not int or kmin < 1 or X.shape[0] <= kmin:
            return None, None
        if type(kmax) is not int or kmax < 1 or X.shape[0] < kmax:
            return None, None
        if kmin >= kmax:
            return None, None

        results = []
        d_vars = []
        for k in range(kmin, kmax + 1):
            C, clss = kmeans(X, k, iterations)
            results = results + [(C, clss)]
            var = variance(X, C)
            if k == kmin:
                minimun_var = var
            d_vars = d_vars + [minimun_var - var]
        return results, d_vars
    except Exception:
        return None, None
