#!/usr/bin/env python3
"""0-create_confusion module
contains the function create_confusion_matrix
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """Creates a confusion matrix.
    Args:
        labels: one-hot encodod `numpy.ndarray` of shape (m, classes)
            that contain the correct labels for each data point
            m: `int`, the number of data points
            classes: `int`, the number of classes
        logits: one-hot encodod `numpy.ndarray` of shape (m, classes)
            m: `int`, the number of data points
            classes: `int`, the number of classes

    Returns:
        confusion_matrix: confusion matrix `numpy.ndarray`
        of shape (classes, classes) with row indices representing the
        correct labels and column indices representing the predicted labels
    """
    confusion_matrix = np.matmul(labels.T, logits)
    return confusion_matrix
