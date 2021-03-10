#!/usr/bin/env python3
"""policy gradients"""
import numpy as np
import gym


def policy(matrix, weight):
    """
    Function that computes to policy with a weight of a matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
        matrix representing an observation of the environment
    weight : numpy.ndarray
        initial matrix of random weight

    Returns
    -------
    numpy.ndarray
        policy
    """
    z = np.matmul(matrix, weight)
    exp = np.exp(z)
    return exp / np.sum(exp)


def policy_gradient(state, weight):
    """
    Function that computes the Monte-Carlo policy gradient
    based on a state and a weight matrix.

    Parameters
    ----------
    state : numpy.ndarray
        matrix representing the current observation of the environment
    weight : numpy.ndarray
        matrix of random weight

    Returns
    -------
    action, gradient : tuple
        the action and the gradient
    """
    # Compute the probability
    P = policy(state, weight)

    # Take an action randomly regarding the probability
    action = np.random.choice(len(P[0]), p=P[0])

    # Compute the gradient,
    # save it with reward to be able to update the weights
    # P looks like [P0, P1]; it's an array of arrays,
    # with one row and two columns
    s = P.reshape(-1, 1)
    # to get the softmax matrix
    softmax = np.diagflat(s) - np.matmul(s, s.T)

    # Take the obs for the action taken
    dsoftmax = softmax[action, :]

    # dlog
    dlog = dsoftmax / P[0, action]

    # update the gradient
    grad = np.matmul(state.T, dlog[None, :])

    return action, grad
