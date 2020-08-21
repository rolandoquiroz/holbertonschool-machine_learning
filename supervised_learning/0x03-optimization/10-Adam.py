#!/usr/bin/env python3
"""10-Adam module
contains the function create_Adam_op
"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """Creates the training operation for a neural network
        in tensorflow using the Adam optimization algorithm

    Args:
        loss: `float`, the loss of the network
        alpha: `float`, the learning rate
        beta1: `float`, the weight used for the first moment
        beta2: `float`, the weight used for the second moment
        epsilon: `float`, small number to avoid division by zero

    Returns:
        `Operation` the Adam optimization operation
    """
    optimizer = tf.train.AdamOptimizer(learning_rate=alpha, beta1=beta1,
                                       beta2=beta2, epsilon=epsilon)
    train_op = optimizer.minimize(loss=loss)
    return train_op
