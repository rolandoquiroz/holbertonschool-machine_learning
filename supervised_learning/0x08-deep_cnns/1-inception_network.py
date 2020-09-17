#!/usr/bin/env python3
"""
1-inception_network module
that contains the function inception_network
"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """Builds the inception network as described in Going Deeper with
        Convolutions (2014): https://arxiv.org/pdf/1409.4842.pdf

        Returns:
            the keras model
    """
