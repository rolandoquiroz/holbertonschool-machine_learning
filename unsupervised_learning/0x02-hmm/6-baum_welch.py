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

    F = np.zeros((N, T))
    F[:, 0] = Initial.T * Emission[:, Observation[0]]
    for i in range(1, T):
        for j in range(N):
            F[j, i] = np.sum(Transition[:, j] *
                             F[:, i - 1] *
                             Emission[j, Observation[i]])
    P = np.sum(F[:, -1])
    return P, F


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

    B = np.zeros((N, T))
    B[:, T - 1] = np.ones((N))
    for i in range(T - 2, -1, -1):
        for j in range(N):
            Transitions = Transition[j, :]
            Emissions = Emission[:, Observation[i + 1]]
            B[j, i] = np.sum((Transitions *
                              B[:, i + 1]) *
                             Emissions)
    P = np.sum(Initial[:, 0] *
               Emission[:, Observation[0]] *
               B[:, 0])

    return P, B


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

    while True:
        P_f, alpha = forward(Observations, Emission, Transition, Initial)
        P_b, beta = backward(Observations, Emission, Transition, Initial)

        xi = np.zeros((N, N, T - 1))
        for i in range(T - 1):
            f1 = np.matmul(alpha[:, i].T, Transition)
            f2 = Emission[:, Observations[i + 1]].T
            f3 = beta[:, i + 1]
            deno = np.matmul(f1 * f2, f3)

            for j in range(N):
                f1 = alpha[j, i]
                f2 = Transition[j]
                f3 = Emission[:, Observations[i + 1]].T
                f4 = beta[:, i + 1].T
                nume = f1 * f2 * f3 * f4
                xi[j, :, i] = nume / deno

        gamma = np.sum(xi, axis=1)

        num = np.sum(xi, 2)
        den = np.sum(gamma, axis=1).reshape((-1, 1))
        Transition = num / den

        xi_sum = np.sum(xi[:, :, T - 2], axis=0)
        xi_sum = xi_sum.reshape((-1, 1))
        gamma = np.hstack((gamma, xi_sum))

        deno = np.sum(gamma, axis=1)

        k = 0
        while k < M:
            gamma_i = gamma[:, Observations == k]
            Emission[:, k] = np.sum(gamma_i, axis=1)
            k += 1
        Emission = np.divide(Emission, deno.reshape((-1, 1)))
        if np.isclose(P_f, P_b):
            break

    return Transition, Emission
