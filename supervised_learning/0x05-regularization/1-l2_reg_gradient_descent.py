#!/usr/bin/env python3
"""1-l2_reg_gradient_descent.py module
contains the function l2_reg_gradient_descent
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updates the weights and biases of a neural network using gradient
        descent with L2 regularization. The neural network uses tanh
        activations on each layer except the last, which uses a softmax
        activation. The weights and biases of the network are updated
        in place

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
        lambtha: `float`, is the L2 regularization parameter
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
        dW = np.matmul(dZ, cache["A"+str(i-1)].T)/m
        db = np.sum(dZ, axis=1, keepdims=True)/m
        dW_reg = dW+((lambtha / m) * auxiliar_weights["W" + str(i)])
        weights["W"+str(i)] = auxiliar_weights["W"+str(i)]-alpha*dW_reg
        weights["b"+str(i)] = auxiliar_weights["b"+str(i)]-alpha*db
