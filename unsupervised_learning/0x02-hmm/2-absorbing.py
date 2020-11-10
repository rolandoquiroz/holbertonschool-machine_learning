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
    Pdiag = np.diagonal(P)
    if not np.any(Pdiag == 1):
        return False
    if np.all(Pdiag == 1):
        return True

    i = 0
    while(i < n):
        j = 0
        while(j < n):
            if (i == j) and (i + 1 < len(P)):
                if (P[i + 1][j] == 0) and (P[i][j + 1] == 0):
                    return False
            j += 1
        i += 1

    return True