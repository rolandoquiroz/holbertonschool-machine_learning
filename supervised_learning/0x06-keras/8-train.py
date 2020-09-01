#!/usr/bin/env python3
"""
module 8-train
contains the function train_model
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None,
                early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """Trains a model using mini-batch gradient descent and also
        analyze validation data using early stopping with learning rate decay
        and save the best iteration of the model

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
        early_stopping: `bool`, indicates whether early stopping should be
            used
        patience: `int`, is the patience used for early stopping
        learning_rate_decay: `bool`, indicates whether learning rate decay
            should be used
        alpha: `float`, is the initial learning rate
        decay_rate: `int`, is the decay rate
        save_best: `bool`, indicates whether to save the model after each
            epoch if it is the best
        filepath: `str`, is the file path where the model should be saved
        verbose: `bool`, determines if output should be printed during
            training
        shuffle: `bool`, determines whether to shuffle the batches every
            epoch

    Returns:
        history: `History`, The History object generated after
            training the model
    """
    def scheduler(epoch):
        """Callback to update the learning rate decay

            Args:
                epoch: `int`

            Returns:
                lr: `float`, the updated learning rate
        """
        lr = alpha/(1+decay_rate*epoch)
        return lr

    callbacks = []

    if early_stopping and validation_data:
        EarlyStopping = K.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=patience)
        callbacks.append(EarlyStopping)

    if learning_rate_decay and validation_data:
        LearningRateScheduler = K.callbacks.\
            LearningRateScheduler(schedule=scheduler,
                                  verbose=1)
        callbacks.append(LearningRateScheduler)

    if save_best:
        ModelCheckpoint = K.callbacks.ModelCheckpoint(filepath=filepath,
                                                      save_best_only=True,
                                                      monitor='val_loss',
                                                      mode='min')
        callbacks.append(ModelCheckpoint)

    history = network.fit(x=data, y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=validation_data,
                          verbose=verbose,
                          callbacks=callbacks,
                          shuffle=shuffle)
    return history
