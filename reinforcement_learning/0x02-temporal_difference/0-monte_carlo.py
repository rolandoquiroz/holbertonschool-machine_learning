#!/usr/bin/env python3
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


def generate_episode(policy, env, max_steps=100):
    """Generate full episodes"""
    done, episode = False, []
    while not done:
        state = env.reset()
        for t in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            experience = (state, action, reward, next_state, done)
            episode.append(experience)
            if done:
                break
            if t >= max_steps - 1:
                episode = []
                break
            state = next_state
    return np.array(episode, np.object)


def monte_carlo(env, V, policy,
                episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Function that performs the Monte Carlo algorithm.

    Parameters
    ----------
    env : gym env
        is the openAI environment instance
    V : numpy.ndarray of shape (s,)
        the value estimate
    policy : function
        function that takes in a state and returns the next action to take
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
    """Monte Carlo prediction"""
    first_visit = True
    nS = env.observation_space.n
    discounts = np.logspace(0, max_steps, num=max_steps,
                            base=gamma, endpoint=False)
    alphas = decay_schedule(alpha, 0.01, 0.3, episodes)
    V_ = np.zeros(nS)
    V_ = V
    V_track = np.zeros((episodes, nS))
    for e in range(episodes):
        episode = generate_episode(policy, env, max_steps)
        visited = np.zeros(nS, dtype=np.bool)
        for t, (state, _, reward, _, _) in enumerate(episode):
            if visited[state] and first_visit:
                continue
            visited[state] = True
            n_steps = len(episode[t:])
            G = np.sum(discounts[:n_steps] * episode[t:, 2])
            V_[state] = V_[state] + alphas[e] * (G - V_[state])
        V_track[e] = V_
    # return V.copy(), V_track
    return V_.copy()
