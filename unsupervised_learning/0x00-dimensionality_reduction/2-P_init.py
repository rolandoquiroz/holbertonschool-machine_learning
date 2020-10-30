#!/usr/bin/env python3
"""
module 2-P_int
contains P_init function
"""
import numpy as np


def P_init(X, perplexity):
    """
    Initializes all variables required to calculate the P affinities in t-SNE
    """
    n, _ = X.shape
    EX = np.sum(np.square(X), axis=1)
    D = (np.add(np.add(-2 * np.dot(X, X.T), EX).T, EX))
    D[range(n), range(n)] = 0
    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log2(perplexity)
    return D, P, betas, H
