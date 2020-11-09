#!/usr/bin/env python3
"""
module 6-baum_welch
contains function baum_welch
"""
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Function that performs the Baum-Welch algorithm for a hidden markov model

    Arguments:
        Observation: a numpy.ndarray of shape (T,)
            Contains the index of the observation
                T is the number of observations
        Emission: numpy.ndarray of shape (M, M)
            Contains the emission probability of a specific observation
            given a hidden state
                Emission[i, j] is the probability of observing j given
                    the hidden state i
                N is the number of hidden states
                M is the number of all possible observations
        Transition: 2D numpy.ndarray of shape (M, N)
            Contains the transition probabilities
                Transition[i, j] is the probability of transitioning
                from the hidden state i to j
        Initial: numpy.ndarray of shape (M, 1)
            Contains the probability of starting in a particular hidden state
        iterations: int
            Number of times expectation-maximization should be performed

    Returns:
        Transition, Emission: (Tuple)
            The converged Transition, Emission
        None, None
            On failure
    """
