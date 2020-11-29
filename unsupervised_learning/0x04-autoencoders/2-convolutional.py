#!/usr/bin/env python3
"""
module 2-convolutional
contains function autencoded
"""
import tensorflow.keras as keras


def autencoded(input_dims, filters, latent_dims):
    """Function that creates a convolutional autencoded. The autencoded model
    is compiled using adam optimization and binary cross-entropy loss.
    Each convolution in the encoder uses a kernel size of (3, 3) with
    same padding and relu activation, followed by max pooling of size (2, 2).
    Each convolution in the decoder, except for the last two, uses a filter
    size of (3, 3) with same padding and relu activation, followed by
    upsampling of size (2, 2).
        The second to last convolution use instead use valid padding.
        The last convolution has the same number of filters as the number
            of channels in input_dims with sigmoid activation and
            no upsampling.

    Arguments:
        input_dims: `tuple` of integers containing the dimensions
            of the model input.
        filters: `list` containing the number of filters for each
            convolutional layer in the encoder, respectively.
            the filters should be reversed for the decoder.
        latent_dims: `tuple` of integers containing  the dimensions
            of the latent space representation.

    Returns:
        (encoder, decoder, auto): `tuple`,
            encoder is the encoder model.
            decoder is the decoder model.
            auto is the full autencoded model.
    """
    # Encoder
    inputs = keras.Input(shape=input_dims)

    enco_lay = keras.layers.Conv2D(filters=filters[0],
                                   kernel_size=(3, 3),
                                   padding='same',
                                   activation='relu')(inputs)

    for i in range(1, len(filters)):
        enco_lay = keras.layers.MaxPooling2D(pool_size=(2, 2),
                                             padding='same')(enco_lay)
        enco_lay = keras.layers.Conv2D(filters=filters[i],
                                       kernel_size=(3, 3),
                                       padding='same',
                                       activation='relu')(enco_lay)

    encoded = keras.layers.MaxPooling2D(pool_size=(2, 2),
                                        padding='same')(enco_lay)

    # Decoder
    encoded_inputs = keras.Input(shape=latent_dims)

    decoder_layers = keras.layers.Conv2D(filters=filters[-1],
                                         kernel_size=(3, 3),
                                         padding='same',
                                         activation='relu')(encoded_inputs)
    decoder_layers = keras.layers.UpSampling2D(size=(2, 2))(decoder_layers)

    for i in range(len(filters)-2, 0, -1):
        decoder_layers = keras.layers.Conv2D(filters=filters[i],
                                             kernel_size=(3, 3),
                                             padding='same',
                                             activation='relu')(decoder_layers)
        decoder_layers = keras.layers.UpSampling2D(size=(2, 2))(decoder_layers)

    decoder_layers = keras.layers.Conv2D(filters=filters[0],
                                         kernel_size=(3, 3),
                                         padding='valid',
                                         activation='relu')(decoder_layers)
    decoder_layers = keras.layers.UpSampling2D(size=(2, 2))(decoder_layers)

    decoded = keras.layers.Conv2D(filters=input_dims[-1],
                                  kernel_size=(3, 3),
                                  padding='same',
                                  activation='sigmoid')(decoder_layers)

    encoder = keras.models.Model(inputs=inputs, outputs=encoded)
    decoder = keras.models.Model(inputs=encoded_inputs, outputs=decoded)

    code = encoder(inputs)
    outputs = decoder(code)

    # Autencoder
    auto = keras.models.Model(inputs=inputs, outputs=outputs)
    auto.compile(optimizer='Adam', loss='binary_crossentropy')

    return encoder, decoder, auto
