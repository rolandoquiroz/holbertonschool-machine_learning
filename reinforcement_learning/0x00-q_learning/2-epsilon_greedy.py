#!/usr/bin/env python3
"""function epsilon_greedy"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Function that uses epsilon-greedy to determine the next action.

    Parameters
    ----------
    Q : numpy.ndarray
        the Q-table
    state : [type]
        the current state
    epsilon : float
        the epsilon to use for the calculation

    Notes
    -----
        Sample p with numpy.random.uniform to determine if your algorithm
        should explore or exploit.
        If exploring, pick the next action with numpy.random.randint from
        all possible actions

    Returns
    -------
    next_action_index : [type]
        the next action index
    """
    p = np.random.uniform(0, 1)

    if p < epsilon:
        next_action_index = np.random.randint(Q.shape[1])
    else:
        next_action_index = np.argmax(Q[state, :])

    return next_action_index
