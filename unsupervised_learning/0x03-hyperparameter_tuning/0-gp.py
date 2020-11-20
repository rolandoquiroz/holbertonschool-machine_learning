#!/usr/bin/env python3
"""
module 0-gp
contains class GaussianProcess
"""
import numpy as np


class GaussianProcess:
    """Class that represents a noiseless 1D Gaussian process"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Class constructor: object attributes initialization

        Arguments:
            X_init: numpy.ndarray of shape (t, 1)
                representing the inputs already sampled with
                the black-box function
            Y_init: numpy.ndarray of shape (t, 1)
                representing the outputs of the black-box function
                for each input in X_init
            t: int
                the number of initial samples
            l: int
                the length parameter for the kernel
            sigma_f: float
                the standard deviation given to the output of
                the black-box function

        Public instance attributes:
            X: corresponding to the respective constructor input
            Y: corresponding to the respective constructor input
            l: corresponding to the respective constructor input
            sigma_f: corresponding to the respective constructor input
            K: float
                Representing the current covariance kernel matrix
                for the Gaussian process
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Public instance method that calculates the covariance kernel
        matrix between two matrices using Radial Basis Function (RBF)

        Parameters:
            X1 is a numpy.ndarray of shape (m, 1)
            X2 is a numpy.ndarray of shape (n, 1)

        Returns:
            K: numpy.ndarray of shape (m, n)
            The covariance kernel matrix
        """
        sqdist = (np.sum(X1 ** 2, 1).reshape(-1, 1) +
                  np.sum(X2 ** 2, 1) -
                  2 * np.matmul(X1, X2.T))
        K = self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)
        return K
    