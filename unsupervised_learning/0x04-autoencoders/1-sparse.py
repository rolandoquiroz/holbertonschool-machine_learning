#!/usr/bin/env python3
"""
module 1-sparse
contains function autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """Function that creates a sparse autoencoder. The sparse autoencoder model
    is compiled using adam optimization and binary cross-entropy loss.
    All layers use a relu activation except for the last layer in the decoder,
    which uses sigmoid.

    Arguments:
        input_dims: `int`, dimensions of the model input.
        hidden_layers: `list`, with the number of nodes
            for each hidden layer in the encoder, respectively.
            the hidden layers should be reversed for the decoder.
        latent_dims: `int`, dimensions of the latent space representation.
        lambtha: `float`, the regularization parameter used for
                 L1 regularization on the encoded output.

    Returns:
        encoder, decoder, auto: `tuple`,
            encoder is the encoder model.
            decoder is the decoder model.
            auto is the sparse autoencoder model.
    """
