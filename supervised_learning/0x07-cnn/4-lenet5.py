#!/usr/bin/env python3
"""
4-lenet5 module
contains function lenet5
"""
import tensorflow as tf


def lenet5(x, y):
    """Builds a modified version of the LeNet-5 architecture using tensorflow

    Args:
        x is a tf.placeholder of shape (m, 28, 28, 1) containing the input
        images for the network
            m is the number of images
        y is a tf.placeholder of shape (m, 10) containing the one-hot labels
        for the network

    Returns:
        a tensor for the softmax activated output_layer
        a training operation that utilizes Adam optimization
            (with default hyperparameters)
        a tensor for the loss of the netowrk
        a tensor for the accuracy of the network
    """
    initializer = tf.contrib.layers.variance_scaling_initializer()

    activation = tf.nn.relu

    conv_layer_1 = tf.layers.Conv2D(filters=6,
                                    kernel_size=5,
                                    padding='same',
                                    activation=activation,
                                    kernel_initializer=initializer)(x)

    p_layer_2 = tf.layers.MaxPooling2D(pool_size=[2, 2],
                                       strides=2)(conv_layer_1)

    conv_layer_3 = tf.layers.Conv2D(filters=16, kernel_size=5,
                                    padding='valid', activation=activation,
                                    kernel_initializer=initializer)(p_layer_2)

    p_layer_4 = tf.layers.MaxPooling2D(pool_size=[2, 2],
                                       strides=2)(conv_layer_3)

    # Flattening between conv and dense layers
    flatten_layer = tf.layers.Flatten()(p_layer_4)

    fc_layer_5 = tf.layers.Dense(units=120, activation=activation,
                                 kernel_initializer=initializer)(flatten_layer)

    fc_layer_6 = tf.layers.Dense(units=84,
                                 kernel_initializer=initializer)(fc_layer_5)

    output_layer = tf.layers.Dense(units=10,
                                   kernel_initializer=initializer)(fc_layer_6)

    loss = tf.losses.softmax_cross_entropy(y, output_layer)

    y_pred = tf.nn.softmax(output_layer)

    train_op = tf.train.AdamOptimizer().minimize(loss)

    y_tag = tf.argmax(y, 1)

    y_pred_tag = tf.argmax(output_layer, 1)

    equality = tf.equal(y_tag, y_pred_tag)

    accuracy = tf.reduce_mean(tf.cast(equality, dtype=tf.float32))

    return y_pred, train_op, loss, accuracy
