#!/usr/bin/env python3
"""
module 0-sequential
contains the function build_model
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network with the Keras library with the Sequential
    class: Sequential API

    Args:
        nx: `int`, is the number of input features to the network
        layers: `list`, is a list containing the number of nodes in each
            layer of the network
        activations: `list`, is a list containing the activation functions
            used for each layer of the network
        lambtha: `float`, is the L2 regularization parameter
        keep_prob: is the probability that a node will be kept for dropout

    Returns:
        model: The keras model
    """
    regularizer = K.regularizers.l2(lambtha)
    # Sequential constructor
    model = K.Sequential()
    # You can create a Sequential model incrementally via the add() method
    model.add(K.layers.Dense(units=layers[0], activation=activations[0],
                             kernel_regularizer=regularizer,
                             input_shape=(nx,)))
    for (layer, activation) in zip(layers[1:], activations[1:]):
        model.add(K.layers.Dropout(rate=1-keep_prob))
        model.add(K.layers.Dense(units=layer, activation=activation,
                                 kernel_regularizer=regularizer))
    return model
