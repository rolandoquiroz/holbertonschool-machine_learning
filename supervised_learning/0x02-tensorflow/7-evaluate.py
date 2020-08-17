#!/usr/bin/env python3
"""7-evaluate module
contains the function evaluate
"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """Evaluates the output of a neural network.

    Args:
        X: `numpy.ndarray` that contains the input data to evaluate
        Y: `numpy.ndarray` that contains the one-hot labels for X
        save_path: `str`,  is the location to load the model from

    Returns:
        The network’s prediction, accuracy, and loss, respectively.
    """
    saver = tf.train.import_meta_graph(save_path + ".meta")

    with tf.Session() as session:
        saver.restore(session, save_path)
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        y_pred = tf.get_collection("y_pred")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        # forward propagation
        ans_prediction = session.run(y_pred, feed_dict={x: X, y: Y})
        ans_evaluated_accuracy = session.run(accuracy, feed_dict={x: X, y: Y})
        ans_cost = session.run(loss, feed_dict={x: X, y: Y})

    return ans_prediction, ans_accuracy, ans_cost
