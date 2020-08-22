#!/usr/bin/env python3
"""14-batch_norm module
contains the function batch_norm
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for a neural network in tensorflow

    Args:
        prev: , the activated output of the previous layer
        n: `int`, the number of nodes in the layer to be created
        activation: `str`, is the activation function that should be used on
            the output of the layer

    Returns:
        A: `Tensor`, the activated output for the layer
    """
    initializer = tf.\
        contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    base_layer = tf.layers.Dense(units=n, kernel_initializer=initializer,
                                 name="base_layer")
    X = base_layer(prev)
    print(X.shape)

    mean, variance = tf.nn.moments(X, axes=[0])
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), trainable=True,
                        name="gamma")
    beta = tf.Variable(tf.constant(0.0, shape=[n]), trainable=True,
                       name="beta")
    print(beta.shape)
    epsilon = 1e-8
    Z = tf.nn.batch_normalization(x=X, mean=mean, variance=variance,
                                  offset=beta, scale=gamma,
                                  variance_epsilon=epsilon, name="Z")
    A = activation(Z)

    return A
