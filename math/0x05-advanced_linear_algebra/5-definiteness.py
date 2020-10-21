#!/usr/bin/env python3
"""
module 5-definiteness
contains the function 5-definiteness
"""
import numpy as np


def definiteness(matrix):
    """Calculates the definiteness of a matrix
    Args:
        matrix (numpy.ndarray): with shape (n, n) whose definiteness
            should be calculated
    Returns:
        (str): Positive definite, Positive semi-definite,
            Negative semi-definite, Negative definite,
            or Indefinite if the matrix is positive definite,
            positive semi-definite, negative semi-definite,
            negative definite of indefinite, respectively.
            If matrix does not fit any of the above categories, return None
            If matrix is not a valid matrix, return None
    Raises:
        TypeError: If matrix is not a numpy.ndarray
    """
    if type(matrix) is not np.ndarray:
        raise TypeError("matrix must be a numpy.ndarray")

    if matrix.shape[0] != matrix.shape[1]:
        return None

    if any(matrix != matrix.T):
        return None

    w, _ = np.linalg.eig(matrix)

    if all(w > 0):
        return 'Positive definite'
    elif all(w >= 0):
        return 'Positive semi-definite'
    elif all(w < 0):
        return 'Negative definite'
    elif all(w <= 0):
        return 'Negative semi-definite'
    else:
        return 'Indefinite'
