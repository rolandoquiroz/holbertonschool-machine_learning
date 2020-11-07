#!/usr/bin/env python3
"""
module 7-maximization
contains function maximization
"""
import numpy as np


def maximization(X, g):
    """Calculates the maximization step in the EM algorithm for a GMM"""
    if type(X) is not np.ndarray or len(X.shape) is not 2:
        return None, None, None

    if type(g) is not np.ndarray or len(g.shape) is not 2:
        return None, None, None
    n, d = X.shape
    if n != g.shape[1]:
        return None, None, None

    k = g.shape[0]

    pbs = np.sum(g, axis=0)
    tryouts = np.ones((n,))
    if not np.isclose(pbs, tryouts).all():
        return None, None, None

    # initialization
    pi = np.zeros((k,))
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))

    for i in range(k):
        pi[i] = np.sum(g[i]) / n
        m[i] = np.matmul(g[i], X) / np.sum(g[i])
        diff = X - m[i]
        S[i] = np.matmul(g[i] * diff.T, diff) / np.sum(g[i])
    return pi, m, S
