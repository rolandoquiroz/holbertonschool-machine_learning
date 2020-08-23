#!/usr/bin/env python3
"""3-mini_batch module
contains the function train_mini_batch
"""
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """Tains a loaded neural network model using mini-batch gradient descent.

    Args:
        X_train: `numpy.ndarray` of shape (m, 784),
            containing the training data.
            m: `int`, the number of data points.
            784 is the number of input features.
        Y_train: `numpy.ndarray` of shape (m, 10) one hot encoded,
            containing the training labels.
            10 is the number of classes the model should classify.
        X_valid: `numpy.ndarray` of shape (m, 784),
            containing the validation data.
        Y_valid: `numpy.ndarray` of shape (m, 10) one hot encoded,
            containing the validation labels.
        batch_size: `int`, the number of data points in a batch.
        epochs: `int`, the number of times the training should pass through
            the whole dataset.
        load_path: `str`, the path from which to load the model.
        save_path: `str`, the path to where the model should be saved after
            training.

    Returns:
        saved_path: `str`, the path where the model was saved.
    """
    m = X_train.shape[0]
    fetcher = tf.train.import_meta_graph(load_path + ".meta")
    saver = tf.train.Saver()

    ''' With the (mini-batch) batch size defined computes
    the total number of batches (mini-batches) to train:
    total_batches = n_batches = batches = total_stes = iterations
    Note that you have to train at least a (mini-batch) batch
    with size minor that batch size defined'''

    with tf.Session() as session:

        fetcher.restore(session, save_path)
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]

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

                X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)

                if m <= batch_size:
                    batches = 1
                if (m > batch_size and (int(m % batch_size) == 0)):
                    batches = m // batch_size
                if (m > batch_size and (int(m % batch_size) != 0)):
                    batches = (m // batch_size) + 1

                batch = 0
                batch_start = 0
                batch_end = batch_size

                while (batch_end <= m):
                    session.run(train_op, feed_dict={x: X_shuffled[batch_start:batch_end],
                                                     y: Y_shuffled[batch_start:batch_end]})

                    if ((batch != 0) and ((batch + 1) % 100 == 0)):
                        step_cost = session.run(loss,
                                                feed_dict={x: X_shuffled[batch_start:batch_end],
                                                           y: Y_shuffled[batch_start:batch_end]})
                        step_accuracy = session.run(accuracy,
                                                    feed_dict={x: X_shuffled[batch_start:batch_end],
                                                               y: Y_shuffled[batch_start:batch_end]})
                        print('\tStep {}:'.format(batch + 1))
                        print('\t\tCost: {}'.format(step_cost))
                        print('\t\tAccuracy: {}'.format(step_accuracy))

                    batch_start = batch_end
                    if batch + 1 < batches - 1:
                        batch_end += batch_size
                    else:
                        if m <= batch_size:
                            batch_end = m
                        if ((m > batch_size) and (int(m % batch_size) == 0)):
                            batch_end += batch_size
                        if ((m > batch_size) and (int(m % batch_size) != 0)):
                            batch_end += int(m % batch_size)
                    batch +=1           

        saved_path = saver.save(session, save_path)
        return saved_path
