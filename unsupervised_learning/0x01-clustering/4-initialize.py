#!/usr/bin/env python3
"""
module 4-initialize
contains function initialize
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Initializes cluster centroids for K-means"""
    if type(X) is not np.ndarray or len(X.shape) is not 2:
        return None, None, None
    if type(k) is not int or k < 1 or k >= X.shape[0]:
        return None, None, None

    d = X.shape[1]
    pi = np.tile(1/k, (k,))
    m = kmeans(X, k)[0]
    S = np.tile(np.identity(d), (k, 1, 1))
    return pi, m, S
