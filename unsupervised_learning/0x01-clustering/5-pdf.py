#!/usr/bin/env python3
"""
module 5-pdf
contains function pdf
"""
import numpy as np


def pdf(X, m, S):
    """Calculates the probability density function of
    a Gaussian distribution
    """

    if type(X) is not np.ndarray or len(X.shape) is not 2:
        return None
    if type(m) is not np.ndarray or len(m.shape) is not 1:
        return None
    if type(S) is not np.ndarray or len(S.shape) is not 2:
        return None

    _, d = X.shape
    if d != m.shape[0] or d != S.shape[0]:
        return None
    if S.shape[0] != S.shape[1] or d != S.shape[1]:
        return None

    den = (((2 * np.pi) ** d) * np.linalg.det(S))**(1/2)
    expo = (-0.5 * np.sum(np.matmul(np.linalg.inv(S),
            (X.T - m[:, np.newaxis])) *
            (X.T - m[:, np.newaxis]), axis=0))
    ans = np.exp(expo) / den
    ans = np.where(ans < 1e-300, 1e-300, ans)
    return ans
