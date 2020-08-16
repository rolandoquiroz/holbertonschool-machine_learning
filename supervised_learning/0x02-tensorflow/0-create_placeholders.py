#!/usr/bin/env python3
"""0-create_placeholders module
contains the function create_placeholders
"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """Returns two placeholders, x and y, for the neural network.

    Args:
        nx: `int`, the number of feature columns in our data.
        classes: `int`, the number of classes in our classifier.

    Returns:
        x: `placeholder` for the input data to the neural network.
        y: `placeholder` for the one-hot labels for the input data.
    """
    x = tf.placeholder("float", [None, nx], name="x")
    y = tf.placeholder("float", [None, classes], name="y")
    return x, y
