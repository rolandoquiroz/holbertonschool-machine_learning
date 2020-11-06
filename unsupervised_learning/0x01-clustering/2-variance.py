#!/usr/bin/env python3
"""
module 2-variance
contains function variance
"""
import numpy as np


def variance(X, C):
    """Calculates the total intra-cluster variance for a data set

    Parameters
    ----------
    X is a numpy.ndarray of shape (n, d) containing the data set
    C is a numpy.ndarray of shape (k, d) containing the centroid means
        for each cluster

    Returns
    -------
     var, or None on failure
         - var is the total variance
    """
    try:
        Dij = np.sqrt(((X - C[:, np.newaxis])**2).sum(axis=-1))
        min_Dij = np.min(Dij, axis=0)
        s = np.sum(min_Dij ** 2)
        return s

    except Exception:
        return None
