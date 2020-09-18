#!/usr/bin/env python3
"""
module 7-densenet121.py
contains the function densenet121
"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Builds the DenseNet-121 architecture as described in Densely Connected
        Convolutional Networks: https://arxiv.org/pdf/1608.06993.pdf

        Args:
            - growth_rate is the growth rate
            - compression is the compression factor
            - All weights use he normal initialization
            - All convolutions are preceded by Batch Normalization and a
                rectified linear activation (ReLU), respectively

        Returns:
            the keras model
    """
    init = K.initializers.he_normal(seed=None)
    input_layer = K.layers.Input(shape=(224, 224, 3))

    layer = K.layers.BatchNormalization(axis=3)(input_layer)
    layer = K.layers.Activation('relu')(layer)
    layer = K.layers.Conv2D(filters=64,
                            kernel_size=(7, 7),
                            strides=(2, 2),
                            padding='same',
                            kernel_initializer=init)(layer)

    layer = K.layers.MaxPool2D(pool_size=(3, 3),
                               strides=(2, 2),
                               padding='same')(layer)

    layer, nb_filters = dense_block(X=layer,
                                    nb_filters=2*growth_rate,
                                    growth_rate=growth_rate,
                                    layers=6)

    layer, nb_filters = transition_layer(X=layer,
                                         nb_filters=nb_filters,
                                         compression=compression)

    layer, nb_filters = dense_block(X=layer,
                                    nb_filters=growth_rate,
                                    growth_rate=growth_rate,
                                    layers=12)

    layer, nb_filters = transition_layer(X=layer,
                                         nb_filters=nb_filters,
                                         compression=compression)

    layer, nb_filters = dense_block(X=layer,
                                    nb_filters=growth_rate,
                                    growth_rate=growth_rate,
                                    layers=24)

    layer, nb_filters = transition_layer(X=layer,
                                         nb_filters=nb_filters,
                                         compression=compression)

    layer, nb_filters = dense_block(X=layer,
                                    nb_filters=growth_rate,
                                    growth_rate=growth_rate,
                                    layers=16)

    layer = K.layers.AveragePooling2D(pool_size=(7, 7),
                                      padding='same')(layer)

    layer = K.layers.Dense(units=1000,
                           activation='softmax',
                           kernel_initializer=init)(layer)

    model = K.models.Model(inputs=input_layer, outputs=layer)

    return model
