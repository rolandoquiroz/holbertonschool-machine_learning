#!/usr/bin/env python3
"""
module 1-kmeans
contains function kmeans
"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """Performs K-means on a dataset

    Parameters
    ----------
    X : numpy.ndarray of shape (n, d)
        Contains the dataset that will be used for K-means clustering
        n is the number of data points
        d is the number of dimensions for each data point
    k : int
        Positive integer containing the number of clusters
    iterations : int
        Positive integer containing the maximum number of iterations
        that should be performed

    Returns
    -------
    centroids : numpy.ndarray of shape (k, d)
        containing the initialized centroids for each cluster,
        or None on failure
    """
    if type(X) is not np.ndarray or len(X.shape) is not 2:
        return None, None

    n, d = X.shape
    if type(k) is not int or not 0 < k < n:
        return None, None

    if type(iterations) is not int or iterations < 1:
        return None, None

    minimum_value = np.amin(X, axis=0)
    maximum_value = np.amax(X, axis=0)
    C = np.random.uniform(low=minimum_value,
                          high=maximum_value,
                          size=(k, d))
    clss = None
    for _ in range(iterations):
        C_j = np.copy(C)
        D_ij = np.linalg.norm(X[:, None] - C_j, axis=-1)
        clss = np.argmin(D_ij, axis=-1)
        for j in range(k):
            index = np.argwhere(clss == j)
            if not len(index):
                C[j] = np.random.uniform(low=minimum_value, high=maximum_value)
            else:
                C[j] = np.mean(X[index], axis=0)
        if (C_j == C).all():
            return C, clss
    D_ij = np.linalg.norm(X[:, None] - C, axis=-1)
    clss = np.argmin(D_ij, axis=-1)

    return C, clss
