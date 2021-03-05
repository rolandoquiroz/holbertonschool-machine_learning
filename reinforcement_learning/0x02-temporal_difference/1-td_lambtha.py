#!/usr/bin/env python3
"""TD(λ) algorithm"""
import numpy as np
import gym


def decay_schedule(init_value, min_value, decay_ratio,
                   max_steps=100, log_start=-2, log_base=10):
    """Exponentially decaying schedule"""
    decay_steps = int(max_steps * decay_ratio)
    rem_steps = max_steps - decay_steps
    values = np.logspace(log_start, 0, decay_steps,
                         base=log_base, endpoint=True)[::-1]
    values = (values - values.min()) / (values.max() - values.min())
    values = (init_value - min_value) * values + min_value
    values = np.pad(values, (0, rem_steps), 'edge')
    return values


def td_lambtha(env, V, policy, lambtha,
               episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Function that performs the TD(λ) algorithm.

    Parameters
    ----------
    env : gym env
        openAI environment instance
    V : numpy.ndarray of shape (s,)
        the value estimate
    policy : function
        function that takes in a state and returns the next action to take
    lambtha : float
        the eligibility trace factor
    episodes : int, optional
        the total number of episodes to train over, by default 5000
    max_steps : int, optional
        the maximum number of steps per episode, by default 100
    alpha : float, optional
        the learning rate, by default 0.1
    gamma : float, optional
        the discount rate, by default 0.99

    Returns
    -------
    V : numpy.ndarray of shape (s,)
        the updated value estimate
    """
    nS = env.observation_space.n
    V_track = np.zeros((episodes, nS))
    E = np.zeros(nS)
    alphas = decay_schedule(alpha, 0.01, 0.3, episodes)
    for e in range(episodes):
        E.fill(0)
        state, done = env.reset(), False
        while not done:
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            td_target = reward + gamma * V[next_state] * (not done)
            td_error = td_target - V[state]
            E[state] = E[state] + 1
            V = V + alphas[e] * td_error * E
            E = gamma * lambtha * E
            state = next_state
        # V_track[e] = V
        # V_, V_track
    return V
