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


def select_action(state, Q, epsilon):
    """epsilon greedy"""
    if np.random.random() > epsilon:
        return np.argmax(Q[state])
    return np.random.randint(len(Q[state]))


def sarsa_lambtha(env, Q, lambtha,
                  episodes=5000,
                  max_steps=100,
                  alpha=0.1,
                  gamma=0.99,
                  epsilon=1,
                  min_epsilon=0.1,
                  epsilon_decay=0.05):
    """
    Function that hat performs SARSA(λ).

    Parameters
    ----------
    env : gym env
        openAI environment instance
    Q : numpy.ndarray of shape (s,a)
        Q-table
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
    epsilon : int, optional
        the initial threshold for epsilon greedy, by default 1
    min_epsilon : float, optional
        the minimum value that epsilon should decay to, by default 0.1
    epsilon_decay : float, optional
        the decay rate for updating epsilon between episodes, by default 0.05

    Returns
    -------
    Q : numpy.ndarray of shape (s,a)
        the updated Q-table
    """
    replacing_traces = True
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q_track = np.zeros((episodes, nS, nA), dtype=np.float64)
    E = np.zeros((nS, nA), dtype=np.float64)
    alphas = decay_schedule(alpha, 0.01, 0.5, episodes)
    epsilons = decay_schedule(epsilon, min_epsilon, epsilon_decay, episodes)
    for e in range(episodes):
        E.fill(0)
        state, done = env.reset(), False
        action = select_action(state, Q, epsilons[e])
        while not done:
            next_state, reward, done, _ = env.step(action)
            next_action = select_action(next_state, Q, epsilons[e])
            td_target = reward + gamma * Q[next_state][next_action] *\
                (not done)
            td_error = td_target - Q[state][action]
            E[state][action] = E[state][action] + 1
            if replacing_traces:
                E.clip(0, 1, out=E)
            Q = Q + alphas[e] * td_error * E
            E = gamma * lambtha * E
            state, action = next_state, next_action
        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))
    V = np.max(Q, axis=1)
    # pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    # Q, V, pi, Q_track, pi_track
    return Q
