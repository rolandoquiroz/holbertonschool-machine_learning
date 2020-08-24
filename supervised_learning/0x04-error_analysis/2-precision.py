#!/usr/bin/env python3
"""2-precision module
contains the function precision
"""
import numpy as np


def precision(confusion):
    """Calculates the precision for each class in a confusion matrix

    Args:
        confusion: A confusion `numpy.ndarray` of shape (classes, classes)
            where row indices represent the correct labels and column indices
            represent the predicted labels
            classes: `int`, the number of classes

    Returns:
        precision: `numpy.ndarray` of shape (classes,) containing the
            precision of each class
    """
    TP = np.diag(confusion)
    FP = confusion.sum(axis=0) - TP
    precision = TP / (TP + FP)
    return precision
