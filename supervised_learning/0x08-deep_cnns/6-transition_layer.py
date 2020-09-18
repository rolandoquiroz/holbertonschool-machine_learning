#!/usr/bin/env python3
"""
module 5-dense_block
contains the function dense_block
"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """Builds a transition layer as described in Densely Connected
        Convolutional Networks: https://arxiv.org/pdf/1608.06993.pdf

        Args:
            - X is the output from the previous layer
            - nb_filters is an integer representing the number of filters in X
            - compression is the compression factor for the transition layer
            - All weights use he normal initialization
            - All convolutions are preceded by Batch Normalization and a
                rectified linear activation (ReLU), respectively

        Returns:
            - The output of the transition layer and the number of filters
                within the output, respectively
    """

    init = K.initializers.he_normal(seed=None)

    layer = K.layers.BatchNormalization()(X)
    layer = K.layers.Activation('relu')(layer)

    nb_filters = int(nb_filters * compression)

    layer = K.layers.Conv2D(filters=nb_filters,
                            kernel_size=(1, 1),
                            padding='same',
                            kernel_initializer=init)(layer)

    X = K.layers.AveragePooling2D(pool_size=(2, 2),
                                  padding='same')(layer)

    return X, nb_filters
