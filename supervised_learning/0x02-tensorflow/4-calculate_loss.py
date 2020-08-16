#!/usr/bin/env python3
"""4-calculate_loss module
contains the function calculate_loss
"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """Calculates the softmax cross-entropy loss of a prediction.

    Args:
        y: `placeholder` for the labels of the input data.
        y_pred: `Tesor` that contains the network’s predictions

    Returns:
        xentropy: `Tensor`, that contains the the softmax cross-entropy
            loss of the prediction.
    """
    xentropy = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)
    return xentropy
