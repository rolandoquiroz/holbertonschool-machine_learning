#!/usr/bin/env python3
"""3-calculate_accuracy module
contains the function calculate_accuracy
"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """Calculates the accuracy of a prediction.

    Args:
        y: `placeholder` for the labels of the input data.
        y_pred: `Tesor` that contains the network’s predictions

    Returns:
        accuracy: `Tensor`, that contains the decimal accuracy
            of the prediction.
    """
    y_label = tf.argmax(y, 1)
    y_hat = tf.argmax(y_pred, 1)
    equality = tf.equal(y_label, y_hat)
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    return accuracy
