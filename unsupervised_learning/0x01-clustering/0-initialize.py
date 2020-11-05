#!/usr/bin/env python3
"""
module 0-initialize
contains function initialize
"""
import numpy as np


def initialize(X, k):
    """Initializes cluster centroids for K-means

    Parameters
    ----------
    X : numpy.ndarray of shape (n, d)
        Contains the dataset that will be used for K-means clustering
        n is the number of data points
        d is the number of dimensions for each data point
    k : int
        Positive integer containing the number of clusters

    Returns
    -------
    numpy.ndarray of shape (k, d)
        contains the initialized centroids for each cluster
        None on failure
    """
    if type(X) is not np.ndarray or len(X.shape) is not 2:
        return None
    if type(k) is not int or k < 1:
        return None
    d = X.shape[1]
    ans = np.random.uniform(low=np.amin(X, axis=0),
                            high=np.amax(X, axis=0),
                            size=(k, d))
    return ans
