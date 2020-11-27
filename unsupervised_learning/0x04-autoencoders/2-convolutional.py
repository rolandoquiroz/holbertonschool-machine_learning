#!/usr/bin/env python3
"""
module 2-convolutional
contains function autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Function that creates a convolutional autoencoder. The autoencoder model
    is compiled using adam optimization and binary cross-entropy loss.
    Each convolution in the encoder uses a kernel size of (3, 3) with
    same padding and relu activation, followed by max pooling of size (2, 2).
    Each convolution in the decoder, except for the last two, uses a filter
    size of (3, 3) with same padding and relu activation, followed by
    upsampling of size (2, 2).
        The second to last convolution use instead use valid padding.
        The last convolution has the same number of filters as the number
            of channels in input_dims with sigmoid activation and no upsampling

    Arguments:
        input_dims: `tuple` of integers containing the dimensions
                    of the model input.
        filters: `list` containing the number of filters for each
                 convolutional layer in the encoder, respectively.
                 the filters should be reversed for the decoder.
        latent_dims: `tuple` of integers containing  the dimensions
                     of the latent space representation.
        lambtha: `float`, the regularization parameter used for
                 L1 regularization on the encoded output.

    Returns:
        encoder, decoder, auto: `tuple`,
                                encoder is the encoder model.
                                decoder is the decoder model.
                                auto is the convolutional autoencoder model.
    """
