#!/usr/bin/env python3
"""8-RMSProp module that
    contains the function create_RMSProp_op
"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """Creates the training operation for a neural network
        in tensorflow using the RMSProp optimization algorithm

    Args:
        loss: `float`, the loss of the network
        alpha: `float`, the learning rate
        beta2: `float`, the RMSProp weight
        epsilon: `float`, small number to avoid division by zero

    Returns:
        `Operation` the RMSProp optimization operation
    """
    optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha, decay=beta2,
                                          epsilon=epsilon)
    train_op = optimizer.minimize(loss=loss)
    return train_op
