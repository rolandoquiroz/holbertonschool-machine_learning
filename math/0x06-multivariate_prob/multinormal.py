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

    def pdf(self, x):
        """
        Public instance method def pdf that calculates the PDF at a data point
        Args:
            x (numpy.ndarray): with shape (d, 1) contains the data point
            whose PDF should be calculated
                d is the number of dimensions of the Multinomial instance
        Returns:
         The value of the PDF
        """
        if type(x) is not np.ndarray:
            raise TypeError('x must be a numpy.ndarray')

        d = self.cov.shape[0]
        n, one = x.shape
        if len(x.shape) is not 2 or n != d or one != 1:
            raise ValueError('x must have the shape ({}, 1)'.format(d))

        m = self.mean
        c = self.cov

        cov_inv = np.linalg.inv(c)
        expo = (-0.5 * np.matmul(np.matmul((x - m).T, cov_inv), x - self.mean))

        pdf = (np.exp(expo[0][0]) /
               np.sqrt(((2 * np.pi) ** n) * np.linalg.det(c)))

        return pdf
