#!/usr/bin/env python3
"""
module 4-deep_rnn contains
function deep_rnn
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Function that performs forward propagation for a deep RNN
    Arguments:
        rnn_cells is a list of RNNCell instances of length l used for
            the forward propagation
            l is the number of layers
        X is a numpy.ndarray of shape (t, m, i) with the data to be used
            t is the maximum number of time steps
            m is the batch size
            i is the dimensionality of the data
        h_0 is numpy.ndarray of shape (l, m, h) with the initial hidden state
            h is the dimensionality of the hidden state
    Returns: H, Y
        H is a numpy.ndarray containing all of the hidden states
        Y is a numpy.ndarray containing all of the outputs
    """
    t, m, i = X.shape
    l, _, h = h_0.shape

    H = np.zeros((t + 1, l, m, h))
    H[0, :, :, :] = h_0

    Y = []

    for k in range(t):
        for layer in range(l):
            if layer == 0:
                h_next, y = rnn_cells[layer].forward(H[k, layer, :, :], X[k])
            else:
                h_next, y = rnn_cells[layer].forward(H[k, layer, :, :], h_next)
            H[k + 1, layer, :, :] = h_next
        Y.append(y)

    Y = np.array(Y)

    return H, Y
