#!/usr/bin/env python3
"""
module 3-lstm_cell contains
class LSTMCell
"""
import numpy as np


def 𝜎(x):
    """ sigmoid function implementation """
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """softmax function"""
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


class LSTMCell:
    """class LSTMCell represents a gated recurrent unit"""

    def __init__(self, i, h, o):
        """
        Constructor
        Arguments:
            i is the dimensionality of the data
            h is the dimensionality of the hidden state
            o is the dimensionality of the outputs
        Public instance attributes:
            Wf, Wu, Wc, Wo, Wy, bf, bu, bc, bo, and by represent the weights
            and biases of the cell.
            The weights are initialized using a random normal distribution in
            the order listed above and the biases are initialized as zeros.
            The weights are used on the right side for h_x multiplication.
                Wf and bf are for the forget gate
                Wu and bu are for the update gate
                Wc and bc are for the intermediate cell state
                Wo and bo are for the output gate
                Wy and by are for the outputs
        """
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
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
            c_prev is a numpy.ndarray of shape (m, h) containing
                the previous cell state
        Returns:
            h_next, c_next, y
                h_next is the next hidden state
                c_next is the next cell state
                y is the output of the cell
        """
        h_x = np.concatenate((h_prev, x_t), axis=1)
        u_t = 𝜎(np.matmul(h_x, self.Wu) + self.bu)
        f_t = 𝜎(np.matmul(h_x, self.Wf) + self.bf)
        o_t = 𝜎(np.matmul(h_x, self.Wo) + self.bo)
        g = np.tanh(np.matmul(h_x, self.Wc) + self.bc)
        c_next = f_t * c_prev + u_t * g
        h_next = o_t * np.tanh(c_next)
        y_linear = np.matmul(h_next, self.Wy) + self.by
        y = softmax(y_linear)
        return h_next, c_next, y
