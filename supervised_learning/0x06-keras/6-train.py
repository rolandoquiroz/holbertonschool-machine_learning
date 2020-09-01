#!/usr/bin/env python3
"""
module 6-train
contains the function train_model
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None,
                early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """Trains a model using mini-batch gradient descent and also
        analyze validation data using early stopping.

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
        early_stopping: `bool`, is a boolean that indicates whether early
            stopping should be used
        patience: `int`, is the patience used for early stopping
        verbose: `bool`, determines if output should be printed during
            training
        shuffle: `bool`, determines whether to shuffle the batches every
            epoch

    Returns:
        history: `History`, The History object generated after
            training the model
    """
    callbacks = []
    if early_stopping and validation_data:
        EarlyStopping = K.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=patience)
        callbacks.append(EarlyStopping)
    history = network.fit(x=data, y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=validation_data,
                          verbose=verbose,
                          callbacks=callbacks,
                          shuffle=shuffle)
    return history
