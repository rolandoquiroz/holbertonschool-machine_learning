#!/usr/bin/env python3
"""
module 0-pca
contains the pca function
"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset
    Parameters
    ----------
    X : numpy.ndarray of shape (n, d) where
        n is the number of data points
        d is the number of dimensions in each point
    var : float
        fraction of the variance that the PCA
        transformation should maintain
    Returns
    -------
    W : numpy.ndarray of shape (d, nd) where
        nd is the new dimensionality of the transformed X
        weight matrix W maintains var fraction of X‘s original variance
        where nd is the new dimensionality of the transformed X
    """
    _, s, vh = np.linalg.svd(X)
    r = np.argmax(np.cumsum(s) > np.sum(s) * var)
    W = vh[:r + 1].transpose()
    return W
