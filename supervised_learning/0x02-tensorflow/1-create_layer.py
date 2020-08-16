#!/usr/bin/env python3
"""0-create_layer module
contains the function create_layer
"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """Creates the layers for the neural network.

    Args:
        prev: `Tensor`, is the tensor output of the previous layer.
        n: `int`, is the number of nodes in the layer to create.
        activation: is the activation function that the layer should use.

    Returns:
        `Tensor`, the tensor output of the layer.
    """
    init = tf.contrib.layers.\
        variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            name="layer", kernel_initializer=init)
    y = layer(prev)
    return y
