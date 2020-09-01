#!/usr/bin/env python3
"""
module 11-config
contains the functions save_configuration and load_configuration
"""
import tensorflow.keras as K


def save_config(network, filename):
    """Saves a keras model’s configuration in JSON format

    Args:
        network: the model whose configuration should be saved
        filename: `str`, is the path of the file
            that the configuration should be saved to

    Returns:
        None
    """
    with open(filename, 'w') as file:
        file.write(network.to_json())
    return None


def load_config(filename):
    """Loads a keras model’s configuration:

    Args:
        filename: `str`,  is the path of the file
            containing the model’s configuration in JSON format

    Returns:
        the loaded model
    """
    with open(filename, 'r') as file:
        loaded_model = K.models.model_from_json(file.read())
    return loaded_model
