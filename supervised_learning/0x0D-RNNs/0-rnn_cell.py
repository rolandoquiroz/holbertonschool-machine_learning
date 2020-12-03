#!/usr/bin/env python3
"""
module 0-rnn_cell contains
Class RNNCell
"""
import numpy as np


class RNNCell():
    """
    Class RNNCell that represents a cell of a simple RNN:
    """
    def __init__(self, i, h, o):
        """
        Constructor
        Arguments:
            i is the dimensionality of the data
            h is the dimensionality of the hidden state
            o is the dimensionality of the outputs
        Public instance attributes:
            Wh, Wy, bh, by that represent the weights and biases of the cell.
            The weights are initialized using a random normal distribution in
            the order listed above and the biases are initialized as zeros.
            The weights are used on the right side for matrix multiplication
                Wh and bh are for the concatenated hidden state and input data
                Wy and by are for the output
        """
        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Public instance method that performs forward propagation
        for one time step
        Arguments:
            x_t is a numpy.ndarray of shape (m, i) that contains
                the data input for the cell
                m is the batch size for the data
            h_prev is a numpy.ndarray of shape (m, h) containing
                the previous hidden state
        Returns:
            h_next, y
                h_next is the next hidden state
                y is the output of the cell
        """
        # current hidden state
        h_x = np.concatenate((h_prev, x_t), axis=1)
        # update hidden state: Note Wh is used on the right side of np.matmul
        h_next = np.tanh(np.matmul(h_x, self.Wh) + self.bh)
        # compute output of the current state: : Again Wy on the right side
        z = np.matmul(h_next, self.Wy) + self.by
        # y = σ(z) = e^z / (1 + e^(-z))
        y = np.exp(z)/np.sum(np.exp(z), axis=1, keepdims=True)

        return h_next, y
