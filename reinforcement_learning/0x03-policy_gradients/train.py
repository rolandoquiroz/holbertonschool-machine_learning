#!/usr/bin/env python3
"""train"""
import numpy as np
import matplotlib.pyplot as plt
import gym
policy = __import__('policy_gradient').policy
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    Function that implements a full training

    Parameters
    ----------
    env : [type]
        initial environment
    nb_episodes : int
        number of episodes used for training
    alpha : float, optional
        the learning rate, by default 0.000045
    gamma : float, optional
        the discount factor, by default 0.98
    show_result : bool, optional
        Render the environment every 1000 episodes computed., by default False

    Returns
    -------
    list
        all values of the score (sum of all rewards during one episode loop)
    """
    # 4 states 2 actions
    w = np.random.rand(4, 2)
    episode_rewards = []

    for episode in range(nb_episodes):
        state = env.reset()[None, :]
        grads = []
        rewards = []
        score = 0
        while True:
            if show_result and (episode % 1000 == 0):
                env.render()
            action, grad = policy_gradient(state, w)
            next_state, reward, done, info = env.step(action)
            next_state = next_state[None, :]
            grads.append(grad)
            rewards.append(reward)
            score += reward
            state = next_state

            if done:
                break

        for i in range(len(grads)):
            w += alpha * grads[i] * sum([r * gamma ** r
                                         for t, r in enumerate(rewards[i:])])

        episode_rewards.append(score)
        print('{}: {}'.format(episode, score), end='\r', flush=False)

    return episode_rewards
