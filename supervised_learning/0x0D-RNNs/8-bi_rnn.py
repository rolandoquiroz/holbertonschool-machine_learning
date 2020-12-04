#!/usr/bin/env python3
"""
module 8-bi_rnn contains
function bi_rnn
"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """Function that performs forward propagation for a bidirectional RNN
    Arguments:
        bi_cells is an instance of BidirectinalCell used for the
            forward propagation
        X is a numpy.ndarray of shape (t, m, i) with the data to be used
            t is the maximum number of time steps
            m is the batch size
            i is the dimensionality of the data
        h_0 is the initial hidden state in the forward direction, given as
            a numpy.ndarray of shape (m, h)
            h is the dimensionality of the hidden state
        h_t is the initial hidden state in the backward direction, given
            as a numpy.ndarray of shape (m, h)
    Returns:
        H, Y:
            H is a numpy.ndarray containing all of the concatenated hidden
                states
            Y is a numpy.ndarray containing all of the outputs
    """
    t = X.shape[0]
    m, h = h_0.shape

    H_f = np.zeros((t, m, h))
    H_b = np.zeros((t, m, h))

    H_f[0] = h_0
    H_b[t - 1] = h_t

    H_next = h_0
    H_prev = h_t

    for k in range(t):
        H_next = bi_cell.forward(H_next, X[k])
        H_brev = bi_cell.backward(H_prev, X[-k - 1])

        H_f[k] = H_next
        H_b[-k - 1] = H_prev

    H = np.concatenate((H_f, H_b), axis=-1)
    Y = bi_cell.output(H)

    return H, Y
