#!/usr/bin/env python3
"""6-dropout_create_layer module
contains the function dropout_create_layer
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """Creates a tensorflow layer that includes L2 regularization

    Args:
        prev: `Tensor`, that contains the output of the previous layer
        n: `int`, the number of nodes the new layer should contain
        activation: `str`, is the activation function that should be used
            on the layer
        keep_prob: `float`, is the probability that a node will be kept

    Returns:
        y: `Tensor`, the output of the new layer
    """
    initializer = tf.contrib.layers.\
        variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=initializer,
                            name="layer")
    drop = tf.layers.Dropout(rate=keep_prob, name="drop")
    y = drop(layer(prev))
    return y
