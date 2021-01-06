#!/usr/bin/env python3
"""Scaled Dot Product Attention"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Function that calculates the scaled dot product attention.
    https://www.tensorflow.org/tutorials/text/transformer#scaled_dot_product_attention

    Arguments:
        Q: tensor with its last two dimensions as (..., seq_len_q, dk)
            the query matrix
        K: tensor with its last two dimensions as (..., seq_len_v, dk)
            the key matrix
        V: tensor with its last two dimensions as (..., seq_len_v, dv)
            the value matrix
        mask: tensor that can be broadcast into (..., seq_len_q, seq_len_v)
            the optional mask, defaulted to None

    Returns: output, weights
        output: tensor with its last two dimensions as (..., seq_len_q, dv)
            the scaled dot product attention
        weights: tensor with its last two dimensions as (..., seq_len_q,
                                                         seq_len_v)
            the attention weights
    """
    QKT = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = QKT / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, V)

    return output, attention_weights
