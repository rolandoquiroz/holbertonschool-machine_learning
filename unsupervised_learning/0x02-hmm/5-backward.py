#!/usr/bin/env python3
"""
module 5-backward
contains function backward
"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Function that performs the backward algorithm for a hidden markov model

    Arguments:
        Observation: a numpy.ndarray of shape (T,)
            Contains the index of the observation
                T is the number of observations
        Emission: numpy.ndarray of shape (N, M)
            Contains the emission probability of a specific observation
            given a hidden state
                Emission[i, j] is the probability of observing j given
                    the hidden state i
                N is the number of hidden states
                M is the number of all possible observations
        Transition: 2D numpy.ndarray of shape (N, N)
            Contains the transition probabilities
                Transition[i, j] is the probability of transitioning
                from the hidden state i to j
        Initial: numpy.ndarray of shape (N, 1)
            Contains the probability of starting in a particular hidden state

    Returns:
        P, B: (Tuple)
            P is the likelihood of the observations given the model
            B is a numpy.ndarray of shape (N, T)
                Contains the backward path probabilities
                    B[i, j] is the probability of generating the future
                    observations from hidden state i at time j
        None, None
            On failure
    """
