#!/usr/bin/env python3
"""
module 1-regular
contains function regular
"""
import numpy as np


def regular(P):
    """
    Function that determines the steady state probabilities
        of a regular markov chain.

    Arguments:
        P: square 2D numpy.ndarray of shape (n, n)
            Represents the transition matrix
                P[i, j] is the probability of transitioning from state i
                    to state j
                n is the number of states in the markov chain

    Returns:
        numpy.ndarray of shape (1, n)
            Containing the steady state probabilities
        None
            On failure
    """
    if type(P) is not np.ndarray or len(P.shape) is not 2:
        return None
    n, columns = P.shape
    if n != columns:
        return None
    if np.sum(P, axis=1).all() != 1:
        return None
    if not np.all(P):
        return None

    A = np.append(P.T-np.eye(n), np.ones((1, n)), axis=0)
    b = np.append(np.zeros((1, n)), 1)
    ans = np.linalg.solve(np.matmul(A.T, A), np.matmul(A.T, b))
    return ans.reshape(1, n)
