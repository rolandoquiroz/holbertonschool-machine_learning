#!/usr/bin/env python3
"""
module 2-absorbing
contains function absorbing
"""
import numpy as np


def absorbing(P):
    """
    Function that determines if a markov chain is absorbing

    Arguments:
        P: square 2D numpy.ndarray of shape (n, n)
            Represents the transition matrix
                P[i, j] is the probability of transitioning from state i
                    to state j
                n is the number of states in the markov chain

    Returns:
        True
            If it is absorbing
        False
            On failure
    """
    if type(P) is not np.ndarray or len(P.shape) is not 2:
        return False
    n, columns = P.shape
    if n != columns:
        return False
    if np.sum(P, axis=1).all() != 1:
        return False
    D = np.diagonal(P)
    if not np.any(D == 1):
        return False
    if np.all(D == 1):
        return True

    count = np.count_nonzero(D == 1)
    B = P[count:, count:]
    Id = np.eye(B.shape[0])
    try:
        if (np.any(np.linalg.inv(Id - B))):
            return True
    except np.linalg.LinAlgError:
        return False
