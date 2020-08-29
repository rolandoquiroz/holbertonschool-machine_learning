#!/usr/bin/env python3
"""5-dropout_gradient_descent
contains the function dropout_gradient_descent
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Updates the weights of a neural network with Dropout regularization
        using gradient descent. All layers except the last should use the
        tanh activation function. The last layer should use the softmax
        activation function

    Args:
        Y: one-hot encoded `numpy.ndarray` of shape (classes, m)
            that contains the correct labels for the data
            classes: `int` is the number of classes
            m: `int` is the number of data points
        weights: `dict`, is a dictionary of the weights `numpy.ndarray` and
            biases `numpy.ndarray` of the neural network
        cache: `dict`, is a dictionary of the outputs of each layer
            of the neural network
        alpha: `float`, is the learning rate
        keep_prob: `float`, is the probability that a node will be kept
        L: `int`, is the number of layers of the network
    """
    auxiliar_weights = weights.copy()
    m = Y.shape[1]
    for i in range(L, 0, -1):
        A = cache["A"+str(i)]
        if i == L:
            dZ = A-Y
        else:
            W = auxiliar_weights["W"+str(i+1)]
            dZ = np.matmul(W.T, dZ)*(1-A**2)
            dZ *= cache["D"+str(i)]
            dZ /= keep_prob
        dW = np.matmul(dZ, cache["A"+str(i-1)].T)/m
        db = np.sum(dZ, axis=1, keepdims=True)/m
        weights["W"+str(i)] = auxiliar_weights["W"+str(i)]-alpha*dW
        weights["b"+str(i)] = auxiliar_weights["b"+str(i)]-alpha*db
