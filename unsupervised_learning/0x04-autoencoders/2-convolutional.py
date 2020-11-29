#!/usr/bin/env python3
"""
module 2-convolutional
contains function autencoded
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
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
    # This is our input placeholder
    inputs = keras.Input(shape=input_dims)

    enco_lay = keras.layers.Conv2D(filters=filters[0],
                                   kernel_size=(3, 3),
                                   padding='same',
                                   activation='relu')(inputs)

    for fltr in filters[1:]:
        enco_lay = keras.layers.MaxPooling2D(pool_size=(2, 2),
                                             padding='same')(enco_lay)

        enco_lay = keras.layers.Conv2D(filters=fltr,
                                       kernel_size=(3, 3),
                                       padding='same',
                                       activation='relu')(enco_lay)

    # "encoded" is the encoded representation of the input
    encoded = keras.layers.MaxPooling2D(pool_size=(2, 2),
                                        padding='same')(enco_lay)

    # This is our encoded input placeholder
    encoded_inputs = keras.Input(shape=latent_dims)

    deco_lay = keras.layers.Conv2D(filters=filters[-1],
                                   kernel_size=(3, 3),
                                   padding='same',
                                   activation='relu')(encoded_inputs)

    deco_lay = keras.layers.UpSampling2D(size=(2, 2))(deco_lay)

    for fltr in reversed(filters[:-1]):
        deco_lay = keras.layers.Conv2D(filters=fltr,
                                       kernel_size=(3, 3),
                                       padding='same',
                                       activation='relu')(deco_lay)

        deco_lay = keras.layers.UpSampling2D(size=(2, 2))(deco_lay)

    deco_lay = keras.layers.Conv2D(filters=filters[0],
                                   kernel_size=(3, 3),
                                   padding='valid',
                                   activation='relu')(deco_lay)

    deco_lay = keras.layers.UpSampling2D(size=(2, 2))(deco_lay)

    # "decoded" is the lossy reconstruction of the input
    decoded = keras.layers.Conv2D(filters=input_dims[-1],
                                  kernel_size=(3, 3),
                                  padding='same',
                                  activation='sigmoid')(deco_lay)

    # Next model maps an input to its encoded representation
    # also called latent space representation or code
    encoder = keras.models.Model(inputs=inputs, outputs=encoded)
    # Next model maps a encoded representation
    # to its lossy reconstruction of the input
    decoder = keras.models.Model(inputs=encoded_inputs, outputs=decoded)

    code = encoder(inputs)
    outputs = decoder(code)

    # This model maps an input to its reconstruction
    auto = keras.models.Model(inputs=inputs, outputs=outputs)
    auto.compile(optimizer='Adam', loss='binary_crossentropy')

    return encoder, decoder, auto
