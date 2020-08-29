#!/usr/bin/env python3
"""3-l2_reg_create_layer module
contains the function l2_reg_create_layer
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Creates a tensorflow layer that includes L2 regularization

    Args:
        prev: `Tensor`, that contains the output of the previous layer
        n: `int`, the number of nodes the new layer should contain
        activation: `str`, is the activation function that should be used
            on the layer
        lambtha: `float`, is the L2 regularization parameter

    Returns:
        y: `Tensor`, the output of the new layer
    """
    regularizer = tf.contrib.layers.l2_regularizer(scale=lambtha)
    initializer = tf.contrib.layers.\
        variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=initializer,
                            kernel_regularizer=regularizer,
                            name="layer")
    y = layer(prev)
    return y
