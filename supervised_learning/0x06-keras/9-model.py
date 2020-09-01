#!/usr/bin/env python3
"""
module 9-model
contains the functions save_model and load_model
"""
import tensorflow.keras as K


def save_model(network, filename):
    """Saves an entire keras model

    Args:
        network: the model to save
        filename: `str`, is the path of the file
            that the model should be saved to

    Returns:
        None
    """
    network.save(filepath=filename)
    return None


def load_model(filename):
    """Loads an entire keras model

    Args:
        filename: `str`,  is the path of the file
            that the model should be loaded from

    Returns:
        network: the loaded model
    """
    network = K.models.load_model(filepath=filename)
    return network
