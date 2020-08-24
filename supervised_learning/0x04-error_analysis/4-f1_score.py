#!/usr/bin/env python3
"""4-f1_score module
contains the function f1_score
"""
import numpy as np


def f1_score(confusion):
    """Calculates the f1_score for each class in a confusion matrix

    Args:
        confusion: A confusion `numpy.ndarray` of shape (classes, classes)
            where row indices represent the correct labels and column indices
            represent the predicted labels
            classes: `int`, the number of classes

    Returns:
        f1_score: `numpy.ndarray` of shape (classes,) containing the
            f1_score of each class
    """
    TP = np.diag(confusion)
    ACTUAL = confusion.sum(axis=1)
    FN = ACTUAL - TP
    PREDICTED = confusion.sum(axis=0)
    FP = PREDICTED - TP
    f1_score = 2*TP / (2*TP + FP + FN)
    return f1_score
