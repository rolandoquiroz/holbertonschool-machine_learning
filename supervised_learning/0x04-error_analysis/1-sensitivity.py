#!/usr/bin/env python3
"""1-sensitivity module
contains the function sensitivity
"""
import numpy as np


def sensitivity(confusion):
    """Calculates the sensitivity for each class in a confusion matrix

    Args:
        confusion: A confusion `numpy.ndarray` of shape (classes, classes)
            where row indices represent the correct labels and column indices
            represent the predicted labels
            classes: `int`, the number of classes

    Returns:
        sensitivity: `numpy.ndarray` of shape (classes,) containing the
            sensitivity of each class
    """
    TP = np.diag(confusion)
    FN = confusion.sum(axis=1) - np.diag(confusion)
    sensitivity = TP/(TP+FN)
    return sensitivity
