"""0-l2_reg_cost module
contains the function l2_reg_cost
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculates the cost of a neural network with L2 regularization

    Args:
        cost: is the cost of the network without L2 regularization
        lambtha: is the regularization parameter
        weights: is a dictionary of the weights and biases (numpy.ndarray s)
            of the neural network
        L: is the number of layers in the neural network
        m: is the number of data points used

    Returns:
        cost of the network accounting for L2 regularization
    """
    Frobenius_norm = 0
    for layer in range(1, L + 1):
        Frobenius_norm += np.linalg.norm(weights['W{}'.format(layer)])
    l2_regularization_cost = cost + (lambtha/(2*m)) * Frobenius_norm
    return l2_regularization_cost
