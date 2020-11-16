#!/usr/bin/env python3
"""
module 6-baum_welch
contains functions forward, backward, baum_welch
"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Function that performs the forward algorithm for a hidden markov model

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
        P, F: (Tuple)
            P is the likelihood of the observations given the model
            F is a numpy.ndarray of shape (N, T)
                Contains the forward path probabilities
                F[i, j] is the probability of being in hidden state i at time j
                given the previous observations
        None, None
            On failure
    """
    T = Observation.shape[0]
    N = Emission.shape[0]

    F = np.zeros((N, T))
    F[:, 0] = np.multiply(Initial[:, 0], Emission[:, Observation[0]])
    for i in range(1, T):
        F[:, i] = np.multiply(np.matmul(F[:, i - 1], Transition),
                              Emission[:, Observation[i]])
    return F


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
    T = Observation.shape[0]
    N = Emission.shape[0]

    B = np.zeros((N, T))
    B[:, T - 1] = np.ones(N)
    for t in range(T - 2, -1, -1):
        B[:, t] = np.sum(Transition *
                         Emission[:, Observation[t + 1]] *
                         B[:, t + 1], axis=1)

    return B


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Function that performs the Baum-Welch algorithm for a hidden markov model

    Arguments:
        Observations: a numpy.ndarray of shape (T,)
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
    if type(Observations) is not np.ndarray:
        return None, None
    if len(Observations.shape) is not 1:
        return None, None
    T = Observations.shape[0]
    if type(Emission) is not np.ndarray or len(Emission.shape) is not 2:
        return None, None
    N, M = Emission.shape
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

    for _ in range(iterations):
        alpha = forward(Observations, Emission, Transition, Initial)
        beta = backward(Observations, Emission, Transition, Initial)

        xi = np.zeros((N, N, T - 1))
        for i in range(T - 1):
            deno = np.matmul(np.matmul(alpha[:, i].T, Transition) *
                             Emission[:, Observations[i + 1]].T,
                             beta[:, i + 1])

            for j in range(N):
                nume = alpha[j, i] * \
                       Transition[j] * \
                       Emission[:, Observations[i + 1]].T * \
                       beta[:, i + 1].T
                xi[j, :, i] = nume / deno

        gamma = np.sum(xi, axis=1)

        Transition = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

        xi_sum = (np.sum(xi[:, :, T - 2], axis=0)).reshape((-1, 1))
        gamma = np.hstack((gamma, xi_sum))

        deno = np.sum(gamma, axis=1)

        k = 0
        while k < M:
            gamma_i = gamma[:, Observations == k]
            Emission[:, k] = np.sum(gamma_i, axis=1)
            k += 1
        Emission = np.divide(Emission, deno.reshape((-1, 1)))

    return Transition, Emission
