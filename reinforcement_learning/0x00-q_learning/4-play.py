#!/usr/bin/env python3
"""function play"""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy
load_frozen_lake = __import__('0-load_env').load_frozen_lake
q_init = __import__('1-q_init').q_init
train = __import__('3-q_learning').train


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

    Returns
    -------
    reward : [type]
        the total rewards for the episode
    """
    state = env.reset()
    done = False
    env.render()

    for step in range(max_steps):
        action = np.argmax(Q[state, :])
        new_state, reward, done, info = env.step(action)
        env.render()

        if done is True:
            if reward == 1:
                print(reward)
                break
        state = new_state

    env.close()

    return reward
