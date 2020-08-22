#!/usr/bin/env python3
"""12-learning_rate_decay module
contains the function learning_rate_decay
"""
import tensorflow as tf


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
