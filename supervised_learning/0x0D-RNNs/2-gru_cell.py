#!/usr/bin/env python3
"""
module 2-gru_cell contains
class GRUCell
"""
import numpy as np


def 𝜎(x):
    """ sigmoid function implementation """
    return 1 / (1 + np.exp(-x))


class GRUCell:
    """class GRUCell represents a gated recurrent unit"""

    def __init__(self, i, h, o):
        """
        Constructor
        Arguments:
            i is the dimensionality of the data
            h is the dimensionality of the hidden state
            o is the dimensionality of the outputs
        Public instance attributes:
            Wz, Wr, Wh, Wy, bz, br, bh, and by represent the weights and
            biases of the cell.
            The weights are initialized using a random normal distribution in
            the order listed above and the biases are initialized as zeros.
            The weights are used on the right side for matrix multiplication.
                Wz and bz are for the update gate
                Wr and br are for the reset gate
                Wh and bh are for the intermediate hidden state
                Wy and by are for the output
        """
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Public instance method that performs forward propagation
        for one time step. The output of the cell uses a softmax
        activation function
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
        # initial info
        h_x = np.concatenate((h_prev, x_t), axis=1)
        # Update gate: what content to throw away and what info to add
        z_t = 𝜎(np.matmul(h_x, self.Wz) + self.bz)
        # Reset gate: which part of the prev content to the main layer
        r_t = 𝜎(np.matmul(h_x, self.Wr) + self.br)
        # updated info
        h_x = np.concatenate((r_t * h_prev, x_t), axis=1)
        # tanh activation outputting hidden activated state
        g = np.tanh(np.matmul(h_x, self.Wh) + self.bh)
        # update hidden state
        h_next = (1 - z_t) * h_prev + z_t * g
        # compute output of the current state
        y_linear = np.matmul(h_next, self.Wy) + self.by
        # softmax activation
        y = np.exp(y_linear) / np.sum(np.exp(y_linear), axis=1, keepdims=True)
        return h_next, y
