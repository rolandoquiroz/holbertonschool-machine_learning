#!/usr/bin/env python3
"""class SelfAttention"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    class SelfAttention to calculate the attention for machine translation
    based on Bahdanau attention https://arxiv.org/pdf/1409.0473.pdf
    """

    def __init__(self, units):
        """
        Class constructor

        Arguments:
            units: (int)
                the number of hidden units in the alignment model

        Public instance attributes:
            W: Dense layer with units units
                to be applied to the previous decoder hidden state
            U: Dense layer with units units
                to be applied to the encoder hidden states
            V: Dense layer with 1 units
                to be applied to the tanh of the sum of the outputs of W and U
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units=units)
        self.U = tf.keras.layers.Dense(units=units)
        self.V = tf.keras.layers.Dense(units=1)

    def call(self, s_prev, hidden_states):
        """
        Method that calls Attention layer to be initializated

        Arguments:
            s_prev: tensor of shape (batch, units)
                the previous decoder hidden state
            hidden_states: tensor of shape (batch, input_seq_len, units)
                the outputs of the encoder

        Returns:
            context, weights
            context: tensor of shape (batch, units)
                the context vector for the decoder
            weights: tensor of shape (batch, input_seq_len, 1)
                the attention weights
        """
        query_with_time_axis = tf.expand_dims(s_prev, axis=1)
        score = self.V(tf.nn.tanh(self.W(query_with_time_axis) +
                                  self.U(hidden_states)))
        attention = tf.nn.softmax(score, axis=1)
        context = attention * hidden_states
        context = tf.reduce_sum(context, axis=1)

        return context, attention
