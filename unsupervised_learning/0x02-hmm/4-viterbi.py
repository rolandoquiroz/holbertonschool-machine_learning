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
    if type(Observation) is not np.ndarray or len(Observation.shape) is not 1:
        return None, None
    T = Observation.shape[0]
    if type(Emission) is not np.ndarray or len(Emission.shape) is not 2:
        return None, None
    N = Emission.shape[0]
    if type(Transition) is not np.ndarray or len(Transition.shape) is not 2:
        return None, None
    if Transition.shape != (N, N):
        return None, None
    if type(Initial) is not np.ndarray or len(Initial.shape) is not 2:
        return None, None
    if Initial.shape != (N, 1):
        return None, None
    if not np.sum(Emission, axis=1).all():
        return None, None
    if not np.sum(Transition, axis=1).all():
        return None, None
    if not np.sum(Initial) == 1:
        return None, None

    bp, vt = np.zeros((N, T)), np.zeros((N, T))
    vt[:, 0] = Initial.T * Emission[:, Observation[0]]
    for i in range(1, T):
        for j in range(N):
            vt[j, i] = np.max(vt[:, i - 1] *
                              Transition[:, j] *
                              Emission[j, Observation[i]])
            bp[j, i] = np.argmax(vt[:, i - 1] *
                                 Transition[:, j] *
                                 Emission[j, Observation[i]])
    S = np.argmax(vt[:, -1])
    path = [S]
    k = T - 1
    while (k > 0):
        S = int(bp[S, k])
        path.append(S)
        k -= 1

    return path[::-1], np.max(vt[:, -1])
