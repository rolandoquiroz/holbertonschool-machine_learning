#!/usr/bin/env python3
"""
Class MultiNormal
"""
import numpy as np


class MultiNormal(object):
    """
    Class that represents a Multivariate Normal distribution
    """

    def __init__(self, data):
        """
        Constructor function
        Args:
            data (numpy.ndarray): with shape (d, n) contains the data set:
                n (int): number of data points
                d (int): number of dimensions in each data point
        Public instance variables:
            mean (numpy.ndarray): with shape (d, 1) containing the mean of data
            cov (numpy.ndarray): with shape (d, d) containing
                the covariance matrix data
        """

        if type(data) is not np.ndarray or len(data.shape) is not 2:
            raise TypeError('data must be a 2D numpy.ndarray')

        d, n = data.shape
        if n < 2:
            raise ValueError('data must contain multiple data points')

        self.mean = (np.mean(data, axis=1)).reshape(d, 1)

        deviation = data - self.mean
        self.cov = np.matmul(deviation, deviation.T) / (n - 1)
