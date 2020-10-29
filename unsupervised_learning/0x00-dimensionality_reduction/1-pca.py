#!/usr/bin/env python3
"""
module 1-pca
contains the pca function
"""
import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset
    Parameters
    ----------
    X : numpy.ndarray of shape (n, d) where
        n is the number of data points
        d is the number of dimensions in each point
    ndim : int
        the new dimensionality of the transformed X
    Returns
    -------
    T : numpy.ndarray of shape (d, ndim)
        contains the transformed version of X
    """
    X_m = X - np.mean(X, axis=0)
    _, _, vh = np.linalg.svd(X_m)
    W = vh[:ndim].transpose()
    T = np.matmul(X_m, W)
    return T
