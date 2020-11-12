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
    if type(kmin) is not int or kmin <= 0 or kmin >= n:
        return None, None, None, None
    if type(kmax) is not int or kmax <= 0 or kmax >= n:
        return None, None, None, None
    if kmin >= kmax:
        return None, None, None, None
    if type(iterations) is not int or iterations < 1:
        return None, None, None, None
    if type(tol) is not float or tol <= 0:
        return None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None

    b, results, ks, l_b = [], [], [], []

    for k in range(kmin, kmax + 1):
        ks.append(k)
        pi, m, S, _, lklhd = expectation_maximization(X,
                                                      k,
                                                      iterations,
                                                      tol,
                                                      verbose)
        results.append((pi, m, S))
        l_b.append(lklhd)
        p = (k * d * (d + 1) / 2) + (d * k) + k - 1
        bic = p * np.log(n) - 2 * lklhd
        b.append(bic)

    l_b = np.array(l_b)
    b = np.array(b)
    i = np.argmin(b)

    return ks[i], results[i], l_b, b
