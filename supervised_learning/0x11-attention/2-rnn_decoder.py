#!/usr/bin/env python3
"""class RNNDecoder"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    class RNNDecoder to decode for machine translation
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor

        Arguments:
            vocab: (int)
                the size of the input vocabulary
            embedding: (int)
                the dimensionality of the embedding vector
            units: (int)
                the number of hidden units in the RNN cell
            batch: (int)
                the batch size

        Public instance attributes:
            embedding: a keras Embedding layer
                converts words from the vocabulary into an embedding vector
            gru: a keras GRU layer with units units
                Recurrent weights initialized with glorot_uniform
                Return the full sequence of outputs and the last hidden state
            F: a Dense layer
                Fully connected dense layer with vocab units
        """
        super(RNNDecoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units=self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.F = tf.keras.layers.Dense(units=vocab)

    def call(self, x, s_prev, hidden_states):
        """
        Arguments:
            x: tensor of shape (batch, 1)
                the previous word in the target sequence as an index
                of the target vocabulary
            s_prev: tensor of shape (batch, units)
                the previous decoder hidden state
            hidden_states: tensor of shape (batch, input_seq_len, units)
                the outputs of the encoder

        Returns: y, s
            y: tensor of shape (batch, vocab)
                the output word as a one hot vector in the target vocabulary
            s: tensor of shape (batch, units)
                the new decoder hidden state
        """
        self.attention = SelfAttention(units=s_prev.shape[1])
        context_vector, attention_weights = self.attention(s_prev,
                                                           hidden_states)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.F(output)
        return x, state
