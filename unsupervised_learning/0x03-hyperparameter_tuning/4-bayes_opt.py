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
        mu, sigma = self.gp.predict(self.X_s)

        '''
        The Multiple-Update-Infill Sampling Method Using
        Minimum Energy Design for Sequential
        Surrogate Modeling by:
        Yongmoon Hwang, Sang-Lyul Cha, Sehoon Kim, Seung-Seop Jin and
        Hyung-Jo Jung

        pag 6:

        the improvement on the minimum can be defined as I(x) = ymin − Y(x)
        the improvement on maximum can be defined by I(x) = Y(x) − ymax
        '''

        if self.minimize is True:
            mu_sample_opt = np.min(self.gp.Y)
            improvement = mu_sample_opt - mu
        else:
            mu_sample_opt = np.max(self.gp.Y)
            improvement = mu - mu_sample_opt

        improvement = improvement - self.xsi

        # EI Loop implementation : Algorithm
        '''
        ac_samples = sigma.shape[0]
        Z, EI = np.zeros(ac_samples)

        for i in range(sigma.shape[0]):
            # if σ(x)>0 : Z = (μ(x)−f(x+)−ξ) / σ(x)
            if sigma[i] > 0:
                Z[i] = improvement[i] / sigma[i]
            # if σ(x)=0 : Z = 0
            else:
                Z[i] = 0
            EI[i] = improvement[i] * norm.cdf(Z[i]) + sigma[i] * norm.pdf(Z[i])
        '''

        # EI Vectorized implementation: Efficient
        with np.errstate(divide='ignore'):
            Z = improvement / sigma
            EI = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            EI[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI
