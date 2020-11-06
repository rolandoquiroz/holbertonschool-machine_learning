#!/usr/bin/env python3
"""
module 6-expectation
contains function expectation
"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Calculates the expectation step in the EM algorithm for a GMM"""

    if type(X) is not np.ndarray or len(X.shape) is not 2:
        return None, None
    if type(m) is not np.ndarray or len(m.shape) is not 2:
        return None, None
    if type(S) is not np.ndarray or len(S.shape) is not 3:
        return None, None
    if type(pi) is not np.ndarray or len(pi.shape) is not 1:
        return None, None
    n, d = X.shape
    if d != S.shape[1] or S.shape[1] != S.shape[2]:
        return (None, None)
    if d != m.shape[1] or m.shape[0] != S.shape[0]:
        return (None, None)
    if pi.shape[0] != m.shape[0]:
        return (None, None)
    if not np.isclose(np.sum(pi), 1):
        return None, None
    k = S.shape[0]
    tmp = np.zeros((k, n))
    for i in range(k):
        P = pdf(X, m[i], S[i])
        prior = pi[i]
        tmp[i] = prior * P
    g = tmp / np.sum(tmp, axis=0)
    likelihood = np.sum(np.log(np.sum(tmp, axis=0)))

    return g, likelihood
