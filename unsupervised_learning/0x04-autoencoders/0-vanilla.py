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
    # This is our original inputs placeholder
    enco_ins = keras.Input(shape=(input_dims, ))

    encoder_layers = keras.layers.Dense(units=hidden_layers[0],
                                        activation='relu',
                                        input_shape=(input_dims, ))(enco_ins)

    for layer in hidden_layers[1:]:
        encoder_layers = keras.layers.Dense(units=layer,
                                            activation='relu')(encoder_layers)

    encoder_layers = keras.layers.Dense(units=latent_dims,
                                        activation='relu')(encoder_layers)

    encoder = keras.Model(inputs=enco_ins, outputs=encoder_layers)

    deco_ins = keras.Input(shape=(latent_dims,))

    decoder_layers = keras.layers.Dense(units=hidden_layers[-1],
                                        activation='relu',
                                        input_shape=(latent_dims, ))(deco_ins)

    for layer in reversed(hidden_layers[:-1]):
        decoder_layers = keras.layers.Dense(units=layer,
                                            activation='relu')(decoder_layers)

    decoder_layers = keras.layers.Dense(units=input_dims,
                                        activation='sigmoid')(decoder_layers)

    decoder = keras.Model(inputs=deco_ins, outputs=decoder_layers)

    ins = keras.Input(shape=(input_dims, ))

    layers = keras.layers.Dense(units=hidden_layers[0],
                                activation='relu',
                                input_shape=(input_dims, ))(ins)

    for layer in hidden_layers[1:]:
        layers = keras.layers.Dense(units=layer,
                                    activation='relu')(layers)

    layers = keras.layers.Dense(units=latent_dims,
                                activation='relu')(layers)

    layers = keras.layers.Dense(units=hidden_layers[-1],
                                activation='relu')(layers)

    for layer in reversed(hidden_layers[:-1]):
        layers = keras.layers.Dense(units=layer,
                                    activation='relu')(layers)

    layers = keras.layers.Dense(units=input_dims,
                                activation='sigmoid')(layers)

    auto = keras.Model(inputs=ins, outputs=layers)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
