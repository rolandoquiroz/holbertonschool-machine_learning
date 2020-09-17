#!/usr/bin/env python3
"""
2-identity_block module
contains the function identity_block
"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """Builds an identity block as described in Deep Residual Learning for
        Image Recognition (2015): https://arxiv.org/pdf/1512.03385.pdf

        Args:
            - A_prev is the output from the previous layer
            - filters: is a tuple or list containing F11, F3, F12,
                respectively:
                - F11: is the number of filters in the 1x1 convolution
                - F3: is the number of filters in the 1x1 convolution before
                    the 3x3 convolution
                - F12: is the number of filters in the 1x1 convolution before
                    the 3x3 convolution
            - All convolutions inside the block should be followed by batch
                normalization along the channels axis and a rectified linear
                activation (ReLU), respectively.
            - All weights should use he normal initialization

        Returns:
            - the activated output of the identity block
    """
    F11, F3, F12 = filters

    initializer = K.initializers.he_normal(seed=None)

    F11_conv = K.layers.Conv2D(filters=F11,
                               kernel_size=(1, 1),
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

    addition = K.layers.Add()([BN_F12_conv,
                               A_prev])

    output = K.layers.Activation('relu')(addition)

    return output
