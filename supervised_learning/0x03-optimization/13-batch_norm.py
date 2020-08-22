#!/usr/bin/env python3
"""13-batch_norm module
contains the function batch_norm
"""


def batch_norm(Z, gamma, beta, epsilon):
    """Normalizes an unactivated output of a neural network using
        batch normalization

    Args:
        Z: `numpy.ndarray` of shape (m, n) to be normalized.
            m: `int`, the number of data points.
            n: `int`, the number of feature columns in Z.
        gamma: `numpy.ndarray`  of shape (1, n) that contains the scales
            used for batch normalization
        beta: `numpy.ndarray`  of shape (1, n) that contains the offsets
            used for batch normalization
        epsilon: `float`, small number used to avoid division by zero

    Returns:
        Z_normalized: `numpy.ndarray`, the normalized Z matrix
    """
    mean = Z.mean(axis=0)
    variance = Z.var(axis=0)
    standard_deviation = (variance + epsilon) ** 0.5
    Z_normalized = (Z - mean) / standard_deviation
    Z_tilde = gamma * Z_normalized + beta
    return Z_tilde
