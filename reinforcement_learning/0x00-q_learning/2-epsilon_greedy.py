#!/usr/bin/env python3
"""function epsilon_greedy"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Function that uses epsilon-greedy to determine the next action.
    Sample p with numpy.random.uniformn to determine if your algorithm
    should explore or exploit.
    If exploring, pick the next action with numpy.random.randint from
    all possible actions

    Parameters
    ----------
    Q : numpy.ndarray
        the q-table
    state : [type]
        the current state
    epsilon : float
        the epsilon to use for the calculation

    Returns
    -------
    next_action_index : [type]
        the next action index
    """
    e = np.random.uniform(0, 1)

    if e < epsilon:
        next_action_index = np.random.randint(Q.shape[1])
    else:
        next_action_index = np.argmax(Q[state, :])

    return next_action_index
