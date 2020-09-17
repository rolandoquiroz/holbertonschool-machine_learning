#!/usr/bin/env python3
"""
4-resnet50 module
contains the function resnet50
"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Builds the ResNet-50 architecture as described in Deep Residual
    Learning for Image Recognition (2015): https://arxiv.org/pdf/1512.03385.pdf

        Returns:
            - the keras model
    """
    init = K.initializers.he_normal(seed=None)

    input_layer = K.Input(shape=(224, 224, 3))

    # conv1
    layer = K.layers.Conv2D(filters=64,
                            kernel_size=(7, 7),
                            strides=(2, 2),
                            padding='same',
                            kernel_initializer=init)(input_layer)
    layer = K.layers.BatchNormalization(axis=3)(layer)
    layer = K.layers.Activation('relu')(layer)

    # conv2_x
    layer = K.layers.MaxPooling2D((3, 3),
                                  padding='same',
                                  strides=(2, 2))(layer)
    layer = projection_block(A_prev=layer, filters=(64, 64, 256), s=1)
    layer = identity_block(A_prev=layer, filters=(64, 64, 256))
    layer = identity_block(A_prev=layer, filters=(64, 64, 256))

    # conv3_x
    layer = projection_block(A_prev=layer, filters=(128, 128, 512))
    layer = identity_block(A_prev=layer, filters=(128, 128, 512))
    layer = identity_block(A_prev=layer, filters=(128, 128, 512))
    layer = identity_block(A_prev=layer, filters=(128, 128, 512))

    # conv4_x
    layer = projection_block(A_prev=layer, filters=(256, 256, 1024))
    layer = identity_block(A_prev=layer, filters=(256, 256, 1024))
    layer = identity_block(A_prev=layer, filters=(256, 256, 1024))
    layer = identity_block(A_prev=layer, filters=(256, 256, 1024))
    layer = identity_block(A_prev=layer, filters=(256, 256, 1024))
    layer = identity_block(A_prev=layer, filters=(256, 256, 1024))

    # conv5_x
    layer = projection_block(A_prev=layer, filters=(512, 512, 2048))
    layer = identity_block(A_prev=layer, filters=(512, 512, 2048))
    layer = identity_block(A_prev=layer, filters=(512, 512, 2048))

    layer = K.layers.AveragePooling2D(pool_size=(7, 7),
                                      padding='same')(layer)

    layer = K.layers.Dense(units=1000,
                           activation='softmax',
                           kernel_initializer=init)(layer)

    model = K.models.Model(inputs=input_layer,
                           outputs=layer)
    return model
