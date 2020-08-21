#!/usr/bin/env python3
"""6-momentum module
contains the function create_momentum_op
"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """Creates the training operation for a neural network
        in tensorflow using the gradient descent with
        momentum optimization algorithm

    Args:
        loss: `float`, the loss of the network
        alpha: `float`, the learning rate
        beta1: `float`, the momentum weight

    Returns:
        `Operation` the momentum optimization operation
    """
    optimizer = tf.train.MomentumOptimizer(learning_rate=alpha, momentum=beta1)
    train_op = optimizer.minimize(loss=loss)
    return train_op
