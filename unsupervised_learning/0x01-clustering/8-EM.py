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
    try:
        if type(tol) is not float or tol < 0:
            return None, None, None, None, None
        if type(verbose) is not bool:
            return None, None, None, None, None
        if iterations < 1:
            return None, None, None, None, None

        pi, m, S = initialize(X, k)
        g, likelihood = expectation(X, pi, m, S)
        likelihood_prev = 0
        for i in range(iterations):
            if i % 10 == 0 and verbose is True:
                print("Log Likelihood after {} iterations: {}"
                      .format(i, likelihood))
            pi, m, S = maximization(X, g)
            g, likelihood = expectation(X, pi, m, S)
            if np.abs(likelihood - likelihood_prev) <= tol:
                if verbose is True:
                    print("Log Likelihood after {} iterations: {}"
                          .format(i + 1, likelihood))
                    break
            likelihood_prev = likelihood
        return pi, m, S, g, likelihood
    except Exception:
        return None, None, None, None, None
