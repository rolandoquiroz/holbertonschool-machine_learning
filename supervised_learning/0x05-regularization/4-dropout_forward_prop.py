#!/usr/bin/env python3
"""4-dropout_forward_prop
contains the function l2_reg_create_layer
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Conducts forward propagation using Dropout.
        All layers except the last should use the
        tanh activation function. The last layer
        should use the softmax activation function

    Args:
        X: numpy.ndarray` of shape (nx, m)
            that contains the input data for the network
            nx: `int` is the number of input features
            m: `int` is the number of data points
        weights: `dict`, is a dictionary of the weights `numpy.ndarray`
            and biases `numpy.ndarray` of the neural network
        L: `int`, is the number of layers of the network
        keep_prob: `float`, is the probability that a node will be kept

    Returns:
        cache: `dict`, a dictionary containing the outputs of each layer
        and the dropout mask used on each layer
    """
    cache = {}
    cache['A0'] = X
    for i in range(L):
        Z = (np.matmul(weights["W{}".format(i+1)],
             cache["A{}".format(i)]) +
             weights["b{}".format(i+1)])
        drop = np.random.binomial(1, keep_prob, size=Z.shape)
        if i == L-1:
            cache["A{}".format(i+1)] = np.exp(Z)/np.sum(np.exp(Z),
                                                        axis=0, keepdims=True)
        else:
            cache["A{}".format(i+1)] = np.tanh(Z)
            cache["D{}".format(i+1)] = drop
            cache["A{}".format(i+1)] = (cache["A{}".format(i+1)] *
                                        cache["D{}".format(i+1)])/keep_prob
    return cache
