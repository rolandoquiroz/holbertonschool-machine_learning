#!/usr/bin/env python3
"""2-l2_reg_cost module
contains the function l2_reg_cost
"""
import tensorflow as tf


def l2_reg_cost(cost):
    """Calculates the cost of a neural network with L2 regularization

    Args:
        cost: `Tensor`, that contains the cost of the network without
            L2 regularization

    Returns:
        J: `Tensor`, that contains the cost of the
            network accounting for L2 regularization
    """
    l2_reg_loss = tf.losses.get_regularization_losses()
    J = cost + l2_reg_loss
    return J
