#!/usr/bin/env python3
"""
module 0-vanilla
contains function autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Function that creates an autoencoder. The autoencoder model is compiled
    using adam optimization and binary cross-entropy loss.
    All layers use a relu activation except for the last layer in the decoder,
    which uses sigmoid.

    Arguments:
        input_dims: `int`, dimensions of the model input.
        hidden_layers: `list`, with the number of nodes:
            for each hidden layer in the encoder, respectively.
            the hidden layers should be reversed for the decoder.
        latent_dims: `int`, dimensions of the latent space representation.

    Returns:
        encoder, decoder, auto: `tuple`,
            encoder is the encoder model.
            decoder is the decoder model.
            auto is the full autoencoder model.
    """
