#!/usr/bin/env python3
"""
module 10-weights
contains the functions save_weights and load_weights
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """Saves a keras model’s weights

    Args:
        network: the model whose weights should be saved
        filename: `str`, is the path of the file
            that the weights should be saved to
        save_format: `str`, is the format in which the
            weights should be saved

    Returns:
        None
    """
    network.save_weights(filepath=filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """Loads a keras model’s weights:

    Args:
        network: the model to which the weights
            should be loaded
        filename: `str`,  is the path of the file
            that the weights should be loaded from

    Returns:
        None
    """
    network.load_weights(filepath=filename)
    return None
