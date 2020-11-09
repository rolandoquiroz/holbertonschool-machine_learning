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
