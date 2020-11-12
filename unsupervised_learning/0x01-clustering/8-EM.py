#!/usr/bin/env python3
"""
module 8-EM
contains function expectation_maximization
"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """Performs the expectation maximization for a GMM"""
    if type(X) is not np.ndarray or len(X.shape) is not 2:
        return None, None, None, None, None
    if type(k) is not int or k <= 0 or k >= X.shape[0]:
        return None, None, None, None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None, None
    if type(tol) is not float or tol <= 0:
        return None, None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    g, lklhd = expectation(X, pi, m, S)
    prev_lklhd = 0

    for i in range(iterations):
        if (i % 10 == 0) and verbose is True:
            print('Log Likelihood after {} iterations: {}'.
                  format(i, lklhd.round(5)))

        pi, m, S = maximization(X, g)
        g, lklhd = expectation(X, pi, m, S)

        if tol >= abs(lklhd - prev_lklhd):
            break
        prev_lklhd = lklhd

    if verbose is True:
        print('Log Likelihood after {} iterations: {}'.
              format(i + 1, lklhd.round(5)))

    return pi, m, S, g, lklhd
