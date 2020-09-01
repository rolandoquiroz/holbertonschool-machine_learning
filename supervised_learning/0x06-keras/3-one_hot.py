#!/usr/bin/env python3
"""
module 3-one_hot
contains the function one_hot
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """Converts a label vector into a one-hot matrix
    Args:
        labels: `numpy.ndarray`, is the list of class labels
        class: `int`, is the maximum number of classes
    Returns:
        one_hot_matrix: `numpy.ndarray`, the one-hot matrix
    """
    one_hot_matrix = K.utils.to_categorical(y=labels, num_classes=classes)
    return one_hot_matrix
