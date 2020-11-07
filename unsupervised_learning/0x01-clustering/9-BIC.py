#!/usr/bin/env python3
"""
module 9-BIC
contains function BIC
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using
    the Bayesian Information Criterion
    """
    if type(X) is not np.ndarray or len(X.shape) is not 2:
        return None, None, None, None
    n, d = X.shape
    if type(kmin) is not int or kmin < 1 or kmin >= n:
        return None, None, None, None
    if type(kmax) is not int or kmax < 1 or kmax >= n:
        return None, None, None, None
    if kmin >= kmax:
        return None, None, None, None
    if type(iterations) is not int or iterations < 1:
        return None, None, None, None
    if type(tol) is not float or tol < 1:
        return None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None

    n, d = X.shape
    k_results, results, l_total, b = [], [], [], []

    for k in range(kmin, kmax + 1):
        pi, m, S, _, like = expectation_maximization(X,
                                                     k,
                                                     iterations,
                                                     tol,
                                                     verbose)
        k_results += [k]
        results += [(pi, m, S)]
        l_total += [like]
        p = (k * d * (d + 1) / 2) + (d * k) + k - 1
        bic = p * np.log(n) - 2 * like
        b += [bic]
    b = np.array(b)
    best = np.argmin(b)
    l_total = np.array(l_total)

    return k_results[best], results[best], l_total[best], b[best]
