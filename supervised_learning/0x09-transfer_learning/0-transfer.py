#!/usr/bin/env python3
"""
0-transfer module trains a convolutional neural network to classify the
CIFAR 10 dataset
contains the function preprocess_data
"""
import tensorflow.keras as K
import tensorflow as tf


def preprocess_data(X, Y):
    """
    Pre-processes the the CIFAR 10 dataset for the VGG16 model

        Args:
            X is a numpy.ndarray of shape (m, 32, 32, 3)
                containing the CIFAR 10 data, where m is
                the number of data points
            Y is a numpy.ndarray of shape (m,) containing
                the CIFAR 10 labels for X

        Returns:
            X_p, Y_p where:
                X_p is a numpy.ndarray containing the preprocessed X
                Y_p is a numpy.ndarray containing the preprocessed Y
    """
    X_p = K.applications.vgg16.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


def preprocess_data(X, Y):
    """
    Pre-processes the the CIFAR 10 dataset for the model

        Args:
            X is a numpy.ndarray of shape (m, 32, 32, 3)
                containing the CIFAR 10 data, where m is
                the number of data points
            Y is a numpy.ndarray of shape (m,) containing
                the CIFAR 10 labels for X

        Returns:
            X_p, Y_p where:
                X_p is a numpy.ndarray containing the preprocessed X
                Y_p is a numpy.ndarray containing the preprocessed Y
    """
    X_p = K.applications.densenet.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == '__main__':
    """
    Trains a CNN based on VGG16 model to classify the CIFAR 10 dataset
    """
    # loads CIFAR 10 dataset
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()

    # pre-procces data
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    # Sets the input for CIFAR 10 Images
    input = K.Input(shape=(32, 32, 3))

    # resizes images to required dimensions for VGG16
    scaled_up_layer = K.layers.Lambda(lambda i: tf.image.resize_image_with_pad(
        image=i,
        target_height=48,
        target_width=48,
        method=tf.image.ResizeMethod.BILINEAR))(input)

    # loads VGG16 base model from Keras Applications
    base_model = K.applications.VGG16(weights='imagenet',
                                      include_top=False,
                                      input_tensor=scaled_up_layer,
                                      input_shape=(48, 48, 3))

    # Extracts the last layer from third block of VGG16 model
    last = base_model.get_layer('block3_pool').output

    # Freezes the layers of VGG16
    base_model.trainable = False

    # Adds classification layers on top of model
    layer = K.layers.GlobalAveragePooling2D()(last)
    layer = K.layers.BatchNormalization()(layer)
    layer = K.layers.Dense(units=256, activation='relu')(layer)
    layer = K.layers.Dropout(rate=0.6)(layer)
    # CIFAR 10 has 10 classes
    classes = 10
    output = K.layers.Dense(units=classes, activation='softmax')(layer)
    model = K.Model(input, output)

    Adam = K.optimizers.Adam(lr=0.0001)

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam,
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=128,
                        validation_data=(x_test, y_test),
                        epochs=20)

    model.save('cifar10.h5')
