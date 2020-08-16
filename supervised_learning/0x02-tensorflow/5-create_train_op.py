#!/usr/bin/env python3
"""5-create_train_op module
contains the function create_train_op
"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """Creates the training operation for the network.

    Args:
        loss: `Tensor`, is the loss of the network’s prediction.
        alpha: `float`, is the learning rate.

    Returns:
        train_op: `Operation`, that trains the network using gradient descent.
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train_op = optimizer.minimize(loss)
    return train_op
