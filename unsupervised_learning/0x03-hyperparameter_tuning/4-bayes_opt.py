#!/usr/bin/env python3
"""
module 4-bayes_opt
contains class BayesianOptimization
"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Class that performs Bayesian optimization on a noiseless
    1D Gaussian process
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """
        Class constructor: object attributes initialization

        Arguments:
            f: function
                the black-box function to be optimized
            X_init: a numpy.ndarray of shape (t, 1)
                represents the inputs already sampled
                with the black-box function
            Y_init: a numpy.ndarray of shape (t, 1)
                represents the outputs of the black-box function
                for each input in X_init
            t: int
                the number of initial samples
            bounds: tuple of (min, max)
                representing the bounds of the space in which
                to look for the optimal point
            ac_samples: int
                number of samples that should be analyzed during acquisition
            l: int
                length parameter for the kernel
            sigma_f: float
                standard deviation given to the output of the
                black-box function
            xsi: float
                the exploration-exploitation factor for acquisition
            minimize: bool
                determining whether optimization should be performed
                for minimization (True) or maximization (False)

        Public instance attributes:
            f: the black-box function
            gp: an instance of the class GaussianProcess
            X_s: numpy.ndarray of shape (ac_samples, 1)
                containing all acquisition sample points,
                evenly spaced between min and max
            xsi: the exploration-exploitation factor
            minimize: a bool for minimization versus maximization
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(start=bounds[0],
                               stop=bounds[1],
                               num=ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Public instance method that calculates the next best sample location
        Using the Expected Improvement acquisition function

        Returns: X_next, EI
            X_next: numpy.ndarray of shape (1,)
                representing the next best sample point
            EI is a numpy.ndarray of shape (ac_samples,)
                containing the expected improvement of each potential sample
        """
        mu_sample, sigma_sample = self.gp.predict(self.X_s)

        if self.minimize is True:
            Y_sample = np.min(self.gp.Y)
            imp = Y_sample - mu_sample - self.xsi
        else:
            Y_sample = np.max(self.gp.Y)
            imp = mu_sample - Y_sample - self.xsi

        Z = np.zeros(sigma_sample.shape[0])
        for i in range(sigma_sample.shape[0]):
            if sigma_sample[i] > 0:
                Z[i] = imp[i] / sigma_sample[i]
        EI = imp * norm.cdf(Z) + sigma_sample * norm.pdf(Z)
        EI[np.isclose(sigma_sample, 0)] = 0.0
        X_next = self.X_s[np.argmax(EI)]


        return X_next, EI
