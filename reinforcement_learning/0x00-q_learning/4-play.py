#!/usr/bin/env python3
"""function play"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Function that has the trained agent play an episode

    Parameters
    ----------
    env : [type]
        the FrozenLakeEnv instance
    Q : numpy.ndarray
        the Q-table
    max_steps : int, optional
        the maximum number of steps in the episode, by default 100

    Notes
    -----
    Each state of the board is displayed via the console
    The Q-table is always exploited

    Returns
    -------
    reward : [type]
        the total rewards for the episode
    """
    state = env.reset()
    env.render()
    for step in range(max_steps):
        action = np.argmax(Q[state, :])
        state, reward, done, info = env.step(action)
        env.render()
        if done:
            break

    return reward
