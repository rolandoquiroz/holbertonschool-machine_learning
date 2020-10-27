#!/usr/bin/env python3
"""
Module 0-mean_cov
Contains function mean_cov
"""
import numpy as np


def mean_cov(X):
    """
    Function that calculates the mean and covariance of a data set.
    Args:
    X (numpy.ndarray): with shape (n, d) contains the data set:
        n (int): is the number of data points
        d (int): is the number of dimensions in each data point
    Returns:
    mean, cov:
        mean (numpy.ndarray): with shape (1, d) contains
            the mean of the data set
        cov (numpy.ndarray): with shape (d, d) containing
            the covariance matrix of the data set
    """

    if type(X) is not np.ndarray or len(X.shape) is not 2:
        raise TypeError('X must be a 2D numpy.ndarray')

    n = X.shape[0]
    if n < 2:
        raise ValueError('X must contain multiple data points')

    d = X.shape[1]
    mean = np.mean(X, axis=0).reshape(1, d)

    deviaton = X - mean

    cov = np.matmul(deviaton.T, deviaton) / (n - 1)
    return mean, cov
