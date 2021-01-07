#!/usr/bin/env python3
"""class RNNEncoder"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    class RNNEncoder to encode for machine translation
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        class RNNEncoder constructor

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
            batch:batch
                the batch size
            units:untis
                the number of hidden units in the RNN cell
            embedding: a keras Embedding layer
                converts words from the vocabulary into an embedding vector
            gru: a keras GRU layer with units units
                Recurrent weights initialized with glorot_uniform
                Return the full sequence of outputs and the last hidden state
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units=self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def initialize_hidden_state(self):
        """
        Method that initializes the hidden states for the RNN cell
        to a tensor of zeros

        Returns:
            Tensor of shape (batch, units)
                Initialized hidden states
        """
        # initializer = tf.keras.initializers.Zeros()
        # initialized_hidden_states = initializer(shape=(self.batch,
        #                                                self.units))
        initialized_hidden_states = tf.zeros(shape=(self.batch,
                                                    self.units))
        return initialized_hidden_states

    def call(self, x, initial):
        """
        Method to make a 'call' for a RNNEncoder layer forward pass.
        Transformation from inputs to outputs

        Arguments:
            x: tensor of shape (batch, input_seq_len)
                the input to the encoder layer as word indices
                within the vocabulary
            initial: tensor of shape (batch, units)
                the initial hidden state

        Returns: outputs, hidden
            output: tensor of shape (batch, input_seq_len, units)
                the outputs of the encoder
            state: tensor of shape (batch, units)
                the last hidden state of the encoder
        """
        x = self.embedding(x)
        output, state = self.gru(inputs=x,
                                 initial_state=initial)
        return output, state
