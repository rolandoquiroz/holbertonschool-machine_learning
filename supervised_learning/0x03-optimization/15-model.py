#!/usr/bin/env python3
"""15-model module
contains the functions model
"""
import tensorflow as tf
import numpy as np


def create_placeholders(nx, classes):
    """Returns two placeholders, x and y, for the neural network.

    Args:
        nx: `int`, the number of feature columns in our data.
        classes: `int`, the number of classes in our classifier.

    Returns:
        x: `placeholder` for the input data to the neural network.
        y: `placeholder` for the one-hot labels for the input data.
    """
    x = tf.placeholder("float", [None, nx], name="x")
    y = tf.placeholder("float", [None, classes], name="y")
    return x, y


def create_layer(prev, n, activation):
    """Creates the layers for the neural network.

    Args:
        prev: `Tensor`, is the tensor output of the previous layer.
        n: `int`, is the number of nodes in the layer to create.
        activation: is the activation function that the layer should use.

    Returns:
        `Tensor`, the tensor output of the layer.
    """
    init = tf.contrib.layers.\
        variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            name="layer", kernel_initializer=init)
    y = layer(prev)
    return y


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for a neural network in tensorflow

    Args:
        prev: , the activated output of the previous layer
        n: `int`, the number of nodes in the layer to be created
        activation: `str`, is the activation function that should be used on
            the output of the layer

    Returns:
        A: `Tensor`, the activated output for the layer
    """
    if activation is None:
        A = create_layer(prev, n, activation)
        return A

    initializer = tf.\
        contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    base_layer = tf.layers.Dense(units=n, kernel_initializer=initializer,
                                 name="base_layer")
    X = base_layer(prev)

    mean, variance = tf.nn.moments(X, axes=[0])
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), trainable=True,
                        name="gamma")
    beta = tf.Variable(tf.constant(0.0, shape=[n]), trainable=True,
                       name="beta")
    epsilon = 1e-8
    Z = tf.nn.batch_normalization(x=X, mean=mean, variance=variance,
                                  offset=beta, scale=gamma,
                                  variance_epsilon=epsilon, name="Z")
    A = activation(Z)
    return A


def forward_prop(x, layer_sizes=[], activations=[]):
    """Creates the forward propagation graph for the neural network.

    Args:
        x: `placeholder` for the input data.
        layer_sizes: `list` that contains the number of nodes in each
            layer of the network.
        activation: `list` that contains the activation functions for
            each layer of the network.

    Returns:
        y_pred: `Tensor`, the prediction of the network .
    """
    y_pred = create_batch_norm_layer(x, layer_sizes[0], activations[0])
    for i in range(1, len(layer_sizes)):
        y_pred = create_batch_norm_layer(y_pred, layer_sizes[i],
                                         activations[i])
    return y_pred


def calculate_accuracy(y, y_pred):
    """Calculates the accuracy of a prediction.

    Args:
        y: `placeholder` for the labels of the input data.
        y_pred: `Tesor` that contains the network’s predictions

    Returns:
        accuracy: `Tensor`, that contains the decimal accuracy
            of the prediction.
    """
    y_label = tf.argmax(y, 1)
    y_hat = tf.argmax(y_pred, 1)
    equality = tf.equal(y_label, y_hat)
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    return accuracy


def calculate_loss(y, y_pred):
    """Calculates the softmax cross-entropy loss of a prediction.

    Args:
        y: `placeholder` for the labels of the input data.
        y_pred: `Tensor` that contains the network’s predictions

    Returns:
        xentropy: `Tensor`, that contains the the softmax cross-entropy
            loss of the prediction.
    """
    xentropy = tf.losses.softmax_cross_entropy(y, y_pred)
    return xentropy


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Creates a learning rate decay operation in tensorflow
        using inverse time decay

    Args:
        alpha: `float`, the original learning rate
        decay_rate: `float`,the weight used to determine the rate at which
            alpha will decay
        global_step: `int`, the number of passes of gradient descent that
            have elapsed
        decay_step: `int`, the number of passes of gradient descent that
            should occur before alpha is decayed further

    Returns:
        learning_rate: `Operation`, the learning rate decay operation
    """
    learning_rate = tf.train.inverse_time_decay(learning_rate=alpha,
                                                global_step=global_step,
                                                decay_steps=decay_step,
                                                decay_rate=decay_rate,
                                                staircase=True)
    return learning_rate


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """Creates the training operation for a neural network
        in tensorflow using the Adam optimization algorithm

    Args:
        loss: `float`, the loss of the network
        alpha: `float`, the learning rate
        beta1: `float`, the weight used for the first moment
        beta2: `float`, the weight used for the second moment
        epsilon: `float`, small number to avoid division by zero

    Returns:
        `Operation` the Adam optimization operation
    """
    optimizer = tf.train.AdamOptimizer(learning_rate=alpha, beta1=beta1,
                                       beta2=beta2, epsilon=epsilon)
    train_op = optimizer.minimize(loss=loss)
    return train_op


def shuffle_data(X, Y):
    """Shuffles the data points in two matrices the same way.
    Args:
        X: `numpy.ndarray` of shape (m, nx), first array to be shuffled.
            m: `int`, the number of data points.
            nx: `int`, the number of feature columns in our data.
        Y: `numpy.ndarray` of shape (m, ny), second array to be shuffled.
            m: `int`, the same number of data points as in X.
            ny: `int`, the number of feature columns in Y.

    Returns:
        X_shuffled, Y_shuffled: `tuple`, Shuffled X and Y matrices.
    """
    # numpy.random.permutation also return a permuted range
    shufled_rows = np.random.permutation(X.shape[0])
    return X[shufled_rows], Y[shufled_rows]


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """Builds, trains, and saves a neural network model in tensorflow using
        Adam optimization, mini-batch gradient descent, learning rate decay,
        and batch normalization

    Args:
        Data_train: `tuple` containing the training inputs and training
            labels, respectively
        Data_valid: `tuple` containing the validation inputs and validation
            labels, respectively
        layers: `list` containing the number of nodes in each layer of the
            network
        activation: `list` containing the activation functions used for
            each layer of the network
        alpha: `float`, the learning rate
        beta1: `float`, the weight for the first moment of Adam Optimization
        beta2: `float`, the weight for the second moment of Adam Optimization
        epsilon: `float`, a small number used to avoid division by zero
        decay_rate: `float`, the decay rate for inverse time decay of the
            learning rate (the corresponding decay step should be 1)
        batch_size: `int`, the number of data points that should be in a
            mini-batch
        epochs: `int`, the number of times the training should pass through
            the whole dataset
        save_path: `str`, the path where the model should be saved to

    Returns:
        saved_path: `str`, the path where the model was saved
    """
    X_train = Data_train[0]
    Y_train = Data_train[1]
    X_valid = Data_valid[0]
    Y_valid = Data_valid[1]
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layers, activations)
    accuracy = calculate_accuracy(y, y_pred)
    loss = calculate_loss(y, y_pred)

    global_step = tf.Variable(0, trainable=False)
    alpha = learning_rate_decay(alpha, decay_rate, global_step, 1)
    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon)

    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('train_op', train_op)

    m = X_train.shape[0]
    saver = tf.train.Saver()
    initializer = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(initializer)

        ''' With the batch size (mini-batch) defined computes
        the total number of batches (mini-batches) to train:
        total_batches = n_batches = batches = total_steps = iterations.
        Note that you have to train at least a batch (mini-batch)
        with size minor or equal that batch size defined'''

        if m <= batch_size:
            batches = 1
        if m > batch_size and m % batch_size == 0:
            batches = m // batch_size
        if m > batch_size and m % batch_size != 0:
            batches = m // batch_size + 1

        for epoch in range(epochs + 1):

            train_accuracy = session.run(accuracy,
                                         feed_dict={x: X_train, y: Y_train})
            train_cost = session.run(loss, feed_dict={x: X_train, y: Y_train})
            valid_accuracy = session.run(accuracy,
                                         feed_dict={x: X_valid, y: Y_valid})
            valid_cost = session.run(loss, feed_dict={x: X_valid, y: Y_valid})
            print('After {} epochs:'.format(epoch))
            print('\tTraining Cost: {}'.format(train_cost))
            print('\tTraining Accuracy: {}'.format(train_accuracy))
            print('\tValidation Cost: {}'.format(valid_cost))
            print('\tValidation Accuracy: {}'.format(valid_accuracy))

            if epoch < epochs:

                session.run(global_step.assign(epoch))
                session.run(alpha)

                X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)

                for batch in range(batches):

                    batch_start = batch * batch_size

                    if batch < batches - 1:
                        batch_end = batch_start + batch_size
                    else:
                        if m <= batch_size:
                            batch_end = m
                        if m > batch_size and m % batch_size == 0:
                            batch_end = batch_start + batch_size
                        if m > batch_size and m % batch_size != 0:
                            batch_end = batch_start + (m % batch_size)

                    session.run(
                        train_op,
                        feed_dict={x: X_shuffled[batch_start:batch_end],
                                   y: Y_shuffled[batch_start:batch_end]})

                    if (batch + 1) % 100 == 0:
                        step_cost = session.run(
                            loss,
                            feed_dict={x: X_shuffled[batch_start:batch_end],
                                       y: Y_shuffled[batch_start:batch_end]})
                        step_accuracy = session.run(
                            accuracy,
                            feed_dict={x: X_shuffled[batch_start:batch_end],
                                       y: Y_shuffled[batch_start:batch_end]})
                        print('\tStep {}:'.format(batch + 1))
                        print('\t\tCost: {}'.format(step_cost))
                        print('\t\tAccuracy: {}'.format(step_accuracy))

        saved_path = saver.save(session, save_path)
        return saved_path
