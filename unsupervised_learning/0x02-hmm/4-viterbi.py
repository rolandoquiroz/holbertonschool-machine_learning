#!/usr/bin/env python3
"""
module 4-viterbi
contains function viterbi
"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Function that calculates the most likely sequence of hidden states for
    a hidden markov model

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
        path, P: (Tuple)
            path is the a list of length T containing
                the most likely sequence of hidden states
            P is the probability of obtaining the path sequence
        None, None
            On failure
    """
