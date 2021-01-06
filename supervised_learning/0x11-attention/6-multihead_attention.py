#!/usr/bin/env python3
"""class MultiHeadAttention"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    class MultiHeadAttention to perform multi head attention
    https://www.tensorflow.org/tutorials/text/transformer
    https://trungtran.io/2019/04/29/create-the-transformer-with-tensorflow-2-0/
    """

    def __init__(self, dm, h):
        """
        Class constructor

        Arguments:
            dm: int
                the dimensionality of the model
            h: int
                the number of heads
            dm is divisible by h

        Public instance attributes:
            h: the number of heads
            dm: the dimensionality of the model
            depth: the depth of each attention head
            Wq: Dense layer with dm units
                to generate the query matrix
            Wk: Dense layer with dm units
                to generate the key matrix
            Wv: Dense layer with dm units
                to generate the value matrix
            linear: Dense layer with dm units
                to generate the attention output
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.WQ = tf.keras.layers.Dense(units=dm)
        self.WK = tf.keras.layers.Dense(units=dm)
        self.WV = tf.keras.layers.Dense(units=dm)
        self.linear = tf.keras.layers.Dense(units=dm)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size,
                                                     num_heads,
                                                     seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Method to make a 'call' for a MultiHeadAttention layer forward pass.
        Transformation from inputs to outputs

        Arguments:
            Q: tensor of shape (batch, seq_len_q, dk)
                the input to generate the query matrix
            K: tensor of shape (batch, seq_len_v, dk)
                the input to generate the key matrix
            V: tensor of shape (batch, seq_len_v, dv)
                the input to generate the value matrix
            mask is always None

        Returns:
            output, weights
            output: tensor with its last two dimensions as (..., seq_len_q, dm)
                the scaled dot product attention
            weights: tensor with its last three dimensions as (..., h,
                                                               seq_len_q,
                                                               seq_len_v)
                the attention weights
        """
        batch_size = tf.shape(Q)[0]

        Q = self.wq(Q)
        K = self.wk(K)
        V = self.wv(V)

        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        scaled_attention, attention_weights = sdp_attention(Q, K, V, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention, (batch_size,
                                                         -1, self.dm))

        output = self.linear(concat_attention)

        return output, attention_weights
