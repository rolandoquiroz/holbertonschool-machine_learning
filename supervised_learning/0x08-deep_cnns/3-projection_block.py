#!/usr/bin/env python3
"""
3-projection_block module
contains the function projection_block
"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """Builds a projection block as described in Deep Residual Learning for
        Image Recognition (2015): https://arxiv.org/pdf/1512.03385.pdf

        Args:
            - A_prev is the output from the previous layer
            - filters: is a tuple or list containing F11, F3, F12,
                respectively:
                - F11 is the number of filters in the first 1x1 convolution
                - F3 is the number of filters in the 3x3 convolution
                - F12 is the number of filters in the second 1x1 convolution
                    as well as the 1x1 convolution in the shortcut connection
            - s is the stride of the first convolution in both the main path
                and the shortcut connection
            - All convolutions inside the block should be followed by batch
                normalization along the channels axis and a rectified linear
                activation (ReLU), respectively.
            - All weights should use he normal initialization

        Returns:
            - the activated output of the projection block
    """
    F11, F3, F12 = filters

    initializer = K.initializers.he_normal(seed=None)

    F11_conv = K.layers.Conv2D(filters=F11,
                               kernel_size=(1, 1),
                               strides=s,
                               padding='same',
                               kernel_initializer=initializer)(A_prev)

    BN_F11_conv = K.layers.BatchNormalization(axis=3)(F11_conv)

    relu_BN_F11_conv = K.layers.Activation('relu')(BN_F11_conv)

    F3_conv = K.layers.Conv2D(filters=F3, kernel_size=(3, 3),
                              padding='same',
                              kernel_initializer=initializer)(relu_BN_F11_conv)

    BN_F3_conv = K.layers.BatchNormalization(axis=3)(F3_conv)

    relu_BN_F3_conv = K.layers.Activation('relu')(BN_F3_conv)

    F12_conv = K.layers.Conv2D(filters=F12,
                               kernel_size=(1, 1),
                               padding='same',
                               kernel_initializer=initializer)(relu_BN_F3_conv)

    BN_F12_conv = K.layers.BatchNormalization(axis=3)(F12_conv)

    skip_connection = K.layers.Conv2D(filters=F12,
                                      kernel_size=(1, 1),
                                      strides=s,
                                      padding='same',
                                      kernel_initializer=initializer)(A_prev)

    BN_skip_connection = K.layers.BatchNormalization(axis=3)(skip_connection)

    addition = K.layers.Add()([BN_F12_conv,
                               BN_skip_connection])

    output = K.layers.Activation('relu')(addition)

    return output
