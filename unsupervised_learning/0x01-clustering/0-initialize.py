#!/usr/bin/env python3
"""
module 0-initialize
contains function initialize
"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means

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
    centroids : numpy.ndarray of shape (k, d)
        containing the initialized centroids for each cluster,
        or None on failure
    """
    if type(X) is not np.ndarray or len(X.shape) is not 2:
        return None
    if type(k) is not int or k < 1:
        return None

    minimum_value = np.amin(X, axis=0)
    maximum_value = np.amax(X, axis=0)
    d = X.shape[1]
    centroids = np.random.uniform(low=minimum_value,
                                  high=maximum_value,
                                  size=(k, d))
    return centroids
