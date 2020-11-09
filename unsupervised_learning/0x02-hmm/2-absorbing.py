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
