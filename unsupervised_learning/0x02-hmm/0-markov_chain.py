#!/usr/bin/env python3
"""
module 0-markov_chain
contains function markov_chain
"""
import numpy as np


def markov_chain(P, s, t=1):
    """
    Function that determines the probability of a markov chain being in a
        particular state after a specified number of iterations.

    Arguments:
        P: square 2D numpy.ndarray of shape (n, n)
            Represents the transition matrix
                P[i, j] is the probability of transitioning from state i
                    to state j
                n is the number of states in the markov chain
        s: numpy.ndarray of shape (1, n)
            Represents the probability of starting in each state
        t: int
            Number of iterations that the markov chain has been through

    Returns:
        numpy.ndarray of shape (1, n)
            Representing the probability of being in a specific state after t
            iterations.
        None
            On failure
    """
    if type(P) is not np.ndarray or len(P.shape) is not 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if type(s) is not np.ndarray or len(P.shape) is not 2:
        return None
    if s.shape[0] is not 1 or s.shape[1] != P.shape[0]:
        return None
    if type(t) is not int or t < 0:
        return None

    ans = np.matmul(s, np.linalg.matrix_power(P, t))
    return ans
