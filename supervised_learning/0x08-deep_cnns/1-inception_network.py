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
    initializer = K.initializers.he_normal(seed=None)

    input_layer = K.Input(shape=(224, 224, 3))

    layer = K.layers.Conv2D(filters=64,
                            kernel_size=(7, 7),
                            strides=(2, 2),
                            padding='same',
                            activation='relu',
                            kernel_initializer=initializer)(input_layer)

    layer = K.layers.MaxPool2D(pool_size=(3, 3),
                               strides=(2, 2),
                               padding='same')(layer)

    layer = K.layers.Conv2D(filters=64,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            padding='same',
                            activation='relu',
                            kernel_initializer=initializer)(layer)

    layer = K.layers.Conv2D(filters=192,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            activation='relu',
                            kernel_initializer=initializer)(layer)

    layer = K.layers.MaxPool2D(pool_size=(3, 3),
                               strides=(2, 2),
                               padding='same')(layer)

    layer = inception_block(A_prev=layer,
                            filters=(64, 96, 128, 16, 32, 32))

    layer = inception_block(A_prev=layer,
                            filters=(128, 128, 192, 32, 96, 64))

    layer = K.layers.MaxPool2D(pool_size=(3, 3),
                               strides=(2, 2),
                               padding='same')(layer)

    layer = inception_block(A_prev=layer,
                            filters=(192, 96, 208, 16, 48, 64))

    layer = inception_block(A_prev=layer,
                            filters=(160, 112, 224, 24, 64, 64))

    layer = inception_block(A_prev=layer,
                            filters=(128, 128, 256, 24, 64, 64))

    layer = inception_block(A_prev=layer,
                            filters=(112, 144, 288, 32, 64, 64))

    layer = inception_block(A_prev=layer,
                            filters=(256, 160, 320, 32, 128, 128))

    layer = K.layers.MaxPool2D(pool_size=(3, 3),
                               strides=(2, 2),
                               padding='same')(layer)

    layer = inception_block(A_prev=layer,
                            filters=(256, 160, 320, 32, 128, 128))

    layer = inception_block(A_prev=layer,
                            filters=(384, 192, 384, 48, 128, 128))

    layer = K.layers.AveragePooling2D(pool_size=(7, 7),
                                      padding='same')(layer)

    layer = K.layers.Dropout(rate=0.4)(layer)

    layer = K.layers.Dense(units=1000,
                           activation='softmax',
                           kernel_initializer=initializer)(layer)

    output_layer = K.models.Model(inputs=input_layer,
                                  outputs=layer)

    return output_layer
