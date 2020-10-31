#!/usr/bin/env python3
"""contains the posterior function"""
from scipy.stats import beta


def posterior(x, n, p1, p2):
    """
    Calculates the posterior probability that the probability of
    developing severe side effects falls within a specific range
    given the data
    """
    if type(n) is not int or n < 1:
        raise ValueError('n must be a positive integer')

    if type(x) is not int or x < 0:
        raise ValueError('x must be an integer that is ' +
                         'greater than or equal to 0')

    if x > n:
        raise ValueError('x cannot be greater than n')

    if type(p1) is not float or p1 < 0 or p1 > 1:
        raise ValueError('p1 must be a float in the range [0, 1]')

    if type(p2) is not float or p2 < 0 or p2 > 1:
        raise ValueError('p2 must be a float in the range [0, 1]')

    if p2 <= p1:
        raise ValueError('p2 must be greater than p1')

    cdf_beta1 = beta.cdf(p1, x + 1, n - x + 1)
    cdf_beta2 = beta.cdf(p2, x + 1, n - x + 1)
    my_posterior = cdf_beta2 - cdf_beta1
    return my_posterior
