#!/usr/bin/env python3
"""
module 1-rnn contains
function rnn
"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """Function that performs forward propagation for a simple RNN
    Arguments:
        rnn_cell is a RNNCell instance used for the forward propagation
        X is a numpy.ndarray of shape (t, m, i) with the data to be used
            t is the maximum number of time steps
            m is the batch size
            i is the dimensionality of the data
        h_0 is numpy.ndarray of shape (m, h) with the initial hidden state
            h is the dimensionality of the hidden state
    Returns: H, Y
        H is a numpy.ndarray containing all of the hidden states
        Y is a numpy.ndarray containing all of the outputs
    """
    t, m, i = X.shape

    H = []
    Y = []

    H[0] = h_0

    for k in range(t):
        h_next, y = rnn_cell.forward(H[-1], X[k, :, :])
        H.append(h_next)
        Y.append(y)
    H = np.array(H)
    Y = np.array(Y)

    return H, Y
