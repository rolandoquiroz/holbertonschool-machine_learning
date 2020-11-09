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
