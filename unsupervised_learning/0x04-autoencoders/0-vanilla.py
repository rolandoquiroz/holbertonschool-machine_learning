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
    original_inputs = keras.Input(shape=(input_dims, ))

    encoded = keras.layers.Dense(units=hidden_layers[0],
                                 activation='relu',
                                 input_shape=(input_dims, ))(original_inputs)

    for hidden_layer in hidden_layers[1:]:
        encoded = keras.layers.Dense(units=hidden_layer,
                                     activation='relu')(encoded)

    encoded = keras.layers.Dense(units=latent_dims,
                                 activation='sigmoid')(encoded)

    encoder = keras.Model(inputs=original_inputs, outputs=encoded)

    encoded_ins = keras.Input(shape=(latent_dims,))

    decoder_layer = keras.layers.Dense(units=hidden_layers[-1],
                                       activation='relu',
                                       input_shape=(latent_dims,))(encoded_ins)

    decoded = keras.layers.Dense(units=hidden_layer,
                                 activation='relu')(encoded)

    for hidden_layer in reversed(hidden_layers[1:]):
        decoder_layer = keras.layers.Dense(units=hidden_layer,
                                           activation='relu')(decoder_layer)
        decoded = keras.layers.Dense(units=hidden_layer,
                                     activation='relu')(decoded)

    decoder_layer = keras.layers.Dense(units=input_dims,
                                       activation='sigmoid')(decoder_layer)
    decoded = keras.layers.Dense(units=input_dims,
                                 activation='sigmoid')(decoded)

    decoder = keras.Model(inputs=encoded_ins, outputs=decoder_layer)
    auto = keras.Model(inputs=original_inputs, outputs=decoded)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
