#!/usr/bin/env python3
"""3-specificity module
contains the function specificity
"""
import numpy as np


def specificity(confusion):
    """Calculates the specificity for each class in a confusion matrix

    Args:
        confusion: A confusion `numpy.ndarray` of shape (classes, classes)
            where row indices represent the correct labels and column indices
            represent the predicted labels
            classes: `int`, the number of classes

    Returns:
        specificity: `numpy.ndarray` of shape (classes,) containing the
            specificity of each class
    """
    TP = np.diag(confusion)
    FP = confusion.sum(axis=0) - TP
    FN = confusion.sum(axis=1) - TP
    TN = confusion.sum() - (FP + FN + TP)
    specificity = TN / (TN + FP)
    return specificity
