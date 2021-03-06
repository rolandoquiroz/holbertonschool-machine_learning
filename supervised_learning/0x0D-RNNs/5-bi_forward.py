#!/usr/bin/env python3
"""
module 5-bi_forward contains
class BidirectionalCell
"""
import numpy as np


class BidirectionalCell:
    """ Represents a bidirectional cell of an RNN """

    def __init__(self, i, h, o):
        """Constructor
        Arguments:
            i is the dimensionality of the data
            h is the dimensionality of the hidden states
            o is the dimensionality of the outputs
        Public instance attributes:
            Whf, Whb, Wy, bhf, bhb, and by represent the weights and biases of
            the cell.
            The weights are initialized using a random normal distribution in
            the order listed above and the biases are initialized as zeros.
            The weights are used on the right side for h_x multiplication.
                Whf and bhf are for the hidden states in the forward direction
                Whb and bhb are for the hidden states in the backward direction
                Wy and by are for the outputs
        """
        self.Whf = np.random.normal(size=(h + i, h))
        self.Whb = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(2 * h, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Public instance method that calculates the hidden state in the forward
        direction for one time step
        Arguments:
            x_t is a numpy.ndarray of shape (m, i) that contains
                the data input for the cell
                m is the batch size for the data
            h_prev is a numpy.ndarray of shape (m, h) containing
                the previous hidden state
        Returns:
            h_next
                h_next is the next hidden state
        """
        h_x = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(h_x, self.Whf) + self.bhf)

        return h_next
