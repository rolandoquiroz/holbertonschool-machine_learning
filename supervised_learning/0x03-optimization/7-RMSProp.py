#!/usr/bin/env python3
"""5-momentum module that
    contains the function update_variables_RMSProp
"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """Updates a variable using the RMSProp optimization algorithm

    Args:
        alpha: `float`, the learning rate
        beta2: `float`, the RMSProp weight
        epsilon: `float`, a small number to avoid division by zero
        var: `numpy.ndarray` containing the variable to be updated
        grad: `numpy.ndarray` containing the gradient of var
        s: is the previous second moment of var

    Returns:
        the updated variable and the new moment, respectively
    """
    s = beta2 * s + (1 - beta2) * grad ** 2
    var = var - (alpha / (s ** 0.5 + epsilon)) * grad
    return var, s
