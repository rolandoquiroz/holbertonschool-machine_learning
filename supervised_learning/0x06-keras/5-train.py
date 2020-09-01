#!/usr/bin/env python3
"""
module 5-train
contains the function train_model
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None,
                verbose=True, shuffle=False):
    """Trains a model using mini-batch gradient descent and also
        analyze validation data.

    Args:
        network: is the model to train
        data: is a `numpy.ndarray` of shape (m, nx) containing the input data
        labels: is a one-hot `numpy.ndarray` of shape (m, classes) containing
            the labels of data
        batch_size: `int`, is the size of the batch used for mini-batch
            gradient descent
        epochs: `int`, is the number of passes through data for mini-batch
            gradient descent
        validation_data: `numpy.ndarray`, is the data to validate the model
            with, if not None
        verbose: `bool`, determines if output should be printed during
            training
        shuffle: `bool`, determines whether to shuffle the batches every
            epoch

    Returns:
        history: `History`, The History object generated after
            training the model
    """
    history = network.fit(x=data, y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=validation_data,
                          verbose=verbose,
                          shuffle=shuffle)
    return history
