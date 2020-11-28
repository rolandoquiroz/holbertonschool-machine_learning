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
        (encoder, decoder, auto): `tuple`,
            encoder is the encoder model.
            decoder is the decoder model.
            auto is the full autoencoder model.
    """
    # This is our input placeholder
    inputs = keras.Input(shape=(input_dims, ))

    encoder_layers = keras.layers.Dense(units=hidden_layers[0],
                                        activation='relu')(inputs)

    for layer in hidden_layers[1:]:
        encoder_layers = keras.layers.Dense(units=layer,
                                            activation='relu')(encoder_layers)

    # "encoded" is the encoded representation of the input
    encoded = keras.layers.Dense(units=latent_dims,
                                 activation='relu')(encoder_layers)

    # This is our encoded input placeholder
    encoded_inputs = keras.Input(shape=(latent_dims,))

    decoder_layers = keras.layers.Dense(units=hidden_layers[-1],
                                        activation='relu')(encoded_inputs)

    for layer in reversed(hidden_layers[:-1]):
        decoder_layers = keras.layers.Dense(units=layer,
                                            activation='relu')(decoder_layers)

    # "decoded" is the lossy reconstruction of the input
    decoded = keras.layers.Dense(units=input_dims,
                                 activation='sigmoid')(decoder_layers)

    # Next model maps an input to its encoded representation
    # also called latent space representation
    encoder = keras.Model(inputs=inputs, outputs=encoded)
    # Next model maps a encoded representation
    # to its lossy reconstruction of the input
    decoder = keras.Model(inputs=encoded_inputs, outputs=decoded)

    code = encoder(inputs)
    outputs = decoder(code)

    # This model maps an input to its reconstruction
    auto = keras.Model(inputs=inputs, outputs=outputs)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
