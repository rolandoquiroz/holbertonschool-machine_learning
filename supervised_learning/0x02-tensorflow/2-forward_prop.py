#!/usr/bin/env python3
"""2-forward_prop module
contains the function forward_prop
"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Creates the forward propagation graph for the neural network.

    Args:
        x: `placeholder` for the input data.
        layer_sizes: `list` that contains the number of nodes in each
            layer of the network.
        activation: `list` that contains the activation functions for
            each layer of the network.

    Returns:
        y_hat: `Tensor`, the prediction of the network .
    """
    y_hat = x
    for i in range(len(layer_sizes)):
        y_hat = create_layer(y_hat, layer_sizes[i], activations[i])
    return y_hat
