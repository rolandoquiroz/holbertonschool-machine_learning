#!/usr/bin/env python3
"""
module 0-likelihood
contains function likelihood
"""
import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining this data given various
    hypothetical probabilities of developing severe side effects

    Args:
        x (int): number of patients that develop severe side effects
        n (int): total number of patients observed
        P (1D numpy.ndarray): contains the various hypothetical
            probabilities of developing severe side effects
    Returns:
        (1D numpy.ndarray): Contains the likelihood of obtaining the
            data, x and n, for each probability in P, respectively
    Raises:
        TypeError:
            If P is not a 1D numpy.ndarray
        ValueError:
            If n is not a positive integer
            If x is not an integer that is greater than or equal to 0
            If x is greater than n
            If any value in P is not in the range [0, 1]
    """
    if type(n) is not int or n <= 0:
        raise ValueError('n must be a positive integer')

    if type(n) is not int or x < 0:
        raise ValueError('x must be an integer that is ' +
                         'greater than or equal to 0')

    if x > n:
        raise ValueError('x cannot be greater than n')

    if type(P) is not np.ndarray or len(P.shape) is not 1:
        raise TypeError('P must be a 1D numpy.ndarray')

    if np.any(P < 0) or np.any(P > 1):
        raise ValueError('All values in P must be in the range [0, 1]')

    num = np.math.factorial(n)
    den = np.math.factorial(x) * np.math.factorial(n - x)
    cs = num / den
    my_likelihood = cs * (P ** x) * (1 - P) ** (n - x)
    return my_likelihood
