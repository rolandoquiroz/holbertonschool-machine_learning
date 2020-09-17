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
    init = K.initializers.he_normal(seed=None)

    input_layer = K.Input(shape=(224, 224, 3))

    convolution_0 = K.layers.Conv2D(filters=64,
                                    kernel_size=(7, 7),
                                    strides=(2, 2),
                                    padding='same',
                                    activation='relu',
                                    kernel_initializer=init)(input_layer)

    max_pool_0 = K.layers.MaxPool2D(pool_size=(3, 3),
                                    strides=(2, 2),
                                    padding='same')(convolution_0)

    convolution_1 = K.layers.Conv2D(filters=64,
                                    kernel_size=(1, 1),
                                    strides=(1, 1),
                                    padding='same',
                                    activation='relu',
                                    kernel_initializer=init)(max_pool_0)

    convolution_2 = K.layers.Conv2D(filters=192,
                                    kernel_size=(3, 3),
                                    strides=(1, 1),
                                    padding='same',
                                    activation='relu',
                                    kernel_initializer=init)(convolution_1)

    max_pool_1 = K.layers.MaxPool2D(pool_size=(3, 3),
                                    strides=(2, 2),
                                    padding='same')(convolution_2)

    inception_3a = inception_block(A_prev=max_pool_1,
                                   filters=(64, 96, 128, 16, 32, 32))

    inception_3b = inception_block(A_prev=inception_3a,
                                   filters=(128, 128, 192, 32, 96, 64))

    max_pool_2 = K.layers.MaxPool2D(pool_size=(3, 3),
                                    strides=(2, 2),
                                    padding='same')(inception_3b)

    inception_4a = inception_block(A_prev=max_pool_2,
                                   filters=(192, 96, 208, 16, 48, 64))

    inception_4b = inception_block(A_prev=inception_4a,
                                   filters=(160, 112, 224, 24, 64, 64))

    inception_4c = inception_block(A_prev=inception_4b,
                                   filters=(128, 128, 256, 24, 64, 64))

    inception_4d = inception_block(A_prev=inception_4c,
                                   filters=(112, 144, 288, 32, 64, 64))

    inception_4e = inception_block(A_prev=inception_4d,
                                   filters=(256, 160, 320, 32, 128, 128))

    max_pool_3 = K.layers.MaxPool2D(pool_size=(3, 3),
                                    strides=(2, 2),
                                    padding='same')(inception_4e)

    inception_5a = inception_block(A_prev=max_pool_3,
                                   filters=(256, 160, 320, 32, 128, 128))

    inception_5b = inception_block(A_prev=inception_5a,
                                   filters=(384, 192, 384, 48, 128, 128))

    avg_pool = K.layers.AveragePooling2D(pool_size=(7, 7),
                                         padding='same')(inception_5b)

    dropout = K.layers.Dropout(rate=0.4)(avg_pool)

    linear = K.layers.Dense(units=1000,
                            activation='softmax',
                            kernel_initializer=init)(dropout)

    the_inception_network = K.models.Model(inputs=input_layer,
                                           outputs=linear)

    return the_inception_network
