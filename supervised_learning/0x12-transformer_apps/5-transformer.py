#!/usr/bin/env python3
"""Transformer class"""
import numy as np
import tensorflow.compat.v2 as tf


def positional_encoding(max_seq_len, dm):
    """
    Function that calculates the positional encoding for a transformer.
    Arguments:
        max_seq_len: int
            the maximum sequence length
        dm: dimension of embedding vector
            the model depth
    Returns:
        PE: numpy.ndarray of shape (max_seq_len, dm)
            positional encoding vectors
    """
    PE = np.zeros([max_seq_len, dm])
    for pos in range(max_seq_len):
        for i in range(0, dm, 2):
            PE[pos, i] = np.sin(pos / (10000 ** (2 * (i // 2) / dm)))
            PE[pos, i + 1] = np.cos(pos / (10000 ** (2 * (i // 2) / dm)))
    return PE


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
    Returns:
        output, weights
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


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    class MultiHeadAttention to perform multi head attention
    https://www.tensorflow.org/tutorials/text/transformer
    https://trungtran.io/2019/04/29/create-the-transformer-with-tensorflow-2-0/
    """

    def __init__(self, dm, h):
        """
        class MultiHeadAttention constructor
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
        self.Wq = tf.keras.layers.Dense(units=dm)
        self.Wk = tf.keras.layers.Dense(units=dm)
        self.Wv = tf.keras.layers.Dense(units=dm)
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

        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        scaled_attention, attention_weights = sdp_attention(Q, K, V, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size,
                                                         -1, self.dm))
        output = self.linear(concat_attention)

        return output, attention_weights


class EncoderBlock(tf.keras.layers.Layer):
    """
    class EncoderBlock to create an encoder block for a transformer
    https://www.tensorflow.org/tutorials/text/transformer
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        class EncoderBlock constructor
        Arguments:
            dm: the dimensionality of the model
            h: the number of heads
            hidden: the number of hidden units in the fully connected layer
            drop_rate: the dropout rate
        Public instance attributes:
            mha: a MultiHeadAttention layer
            dense_hidden: the hidden dense layer with hidden units
                and relu activation
            dense_output: the output dense layer with dm units
            layernorm1: the first layer norm layer, with epsilon=1e-6
            layernorm2: the second layer norm layer, with epsilon=1e-6
            dropout1: the first dropout layer
            dropout2: the second dropout layer
        """
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        Method to make a 'call' for a EncoderBlock layer forward pass.
        Transformation from inputs to outputs
        Arguments:
            x: tensor of shape (batch, input_seq_len, dm)
                the input to the encoder block
            training: boolean
                to determine if the model is training
            mask: the mask to be applied for multi head attention
        Returns:
            a tensor of shape (batch, input_seq_len, dm)
                the block’s output
        """
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.dense_hidden(out1)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class DecoderBlock(tf.keras.layers.Layer):
    """
    class DecoderBlock to create an decoder block for a transformer
    https://www.tensorflow.org/tutorials/text/transformer
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        class DecoderBlock constructor
        Arguments:
            dm: the dimensionality of the model
            h: the number of heads
            hidden: the number of hidden units in the fully connected layer
            drop_drop_rate: the dropout drop_rate
        Public instance attributes:
            mha1 - the first MultiHeadAttention layer
            mha2 - the second MultiHeadAttention layer
            dense_hidden - the hidden dense layer with hidden units
                and relu activation
            dense_output - the output dense layer with dm units
            layernorm1 - the first layer norm layer, with epsilon=1e-6
            layernorm2 - the second layer norm layer, with epsilon=1e-6
            layernorm3 - the third layer norm layer, with epsilon=1e-6
            dropout1 - the first dropout layer
            dropout2 - the second dropout layer
            dropout3 - the third dropout layer
        """
        super(DecoderBlock, self).__init__()

        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)

        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Method to make a 'call' for a DecoderBlock layer forward pass.
        Transformation from inputs to outputs
        Arguments:
            x : a tensor of shape (batch, target_seq_len, dm)
                the input to the decoder block
            encoder_output : a tensor of shape (batch, input_seq_len, dm)
                the output of the encoder
            training : a boolean to determine if the model is training
            look_ahead_mask : the mask to be applied to the first multi
                head attention layer
            padding_mask : the mask to be applied to the second multi
                head attention layer
        Returns:
            a tensor of shape (batch, input_seq_len, dm)
                the block’s output
        """
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(out1,
                                               encoder_output,
                                               encoder_output,
                                               padding_mask)

        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.dense_hidden(out2)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3


class Encoder(tf.keras.layers.Layer):
    """
    class Encoder to create an encoder block for a transformer
    https://www.tensorflow.org/tutorials/text/transformer
    """

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        class Encoder constructor
        Arguments:
            N: the number of blocks in the encoder
            dm: the dimensionality of the model
            h: the number of heads
            hidden: the number of hidden units in the fully connected layer
            input_vocab: the size of the input vocabulary
            max_seq_len: the maximum sequence length possible
            drop_rate: the dropout rate
        Public instance attributes:
            N: the number of blocks in the encoder
            dm: the dimensionality of the model
            embedding: the embedding layer for the inputs
            positional_encoding: a numpy.ndarray of shape (max_seq_len, dm)
                containing the positional encodings
            blocks: a list of length N containing all of the EncoderBlock‘s
            dropout: the dropout layer, to be applied to
                the positional encodings
        """
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Method to make a 'call' for a Encoder layer forward pass.
        Transformation from inputs to outputs
        Arguments:
            x: tensor of shape (batch, input_seq_len, dm)
                the input to the encoder block
            training: boolean
                to determine if the model is training
            mask: the mask to be applied for multi head attention
        Returns:
            a tensor of shape (batch, input_seq_len, dm)
                the block’s output
        """
        seq_len = x.shape[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]
        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.blocks[i](x, training, mask)

        return x


class Decoder(tf.keras.layers.Layer):
    """
    class Decoder to create the decoder for a transformer
    https://www.tensorflow.org/tutorials/text/transformer
    """

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        class Decoder constructor
        Arguments:
            N: the number of blocks in the decoder
            dm: the dimensionality of the model
            h: the number of heads
            hidden: the number of hidden units in the fully connected layer
            target_vocab: the size of the target vocabulary
            max_seq_len: the maximum sequence length possible
            drop_rate: the dropout rate
        Public instance attributes:
            N: the number of blocks in the decoder
            dm: the dimensionality of the model
            embedding: the embedding layer for the inputs
            positional_encoding: a numpy.ndarray of shape (max_seq_len, dm)
                containing the positional encodings
            blocks: a list of length N containing all of the DecoderBlock‘s
            dropout: the dropout layer, to be applied to
                the positional encodings
        """
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask,
             padding_mask):
        """
        Method to make a 'call' for a Decoder layer forward pass.
        Transformation from inputs to outputs
        Arguments:
            x: tensor of shape (batch, target_seq_len, dm)
                the input to the decoder block
            encoder_output: a tensor of shape (batch, input_seq_len, dm)
                the output of the encoder
            training: boolean
                to determine if the model is training
            look_ahead_mask: the mask to be applied to
                the first multi head attention layer
            padding_mask: the mask to be applied to
                the second multi head attention layer
        Returns:
            a tensor of shape (batch, target_seq_len, dm)
                the decoder output
        """
        seq_len = x.shape[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]
        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.blocks[i](x, encoder_output, training,
                               look_ahead_mask, padding_mask)

        return x


class Transformer(tf.keras.Model):
    """
    class Transformer to create a transformer network
    https://www.tensorflow.org/tutorials/text/transformer
    """

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """
        class Transformer constructor
        Arguments:
            N: the number of blocks in the decoder
            dm: the dimensionality of the model
            h: the number of heads
            hidden: the number of hidden units in the fully connected layer
            input_vocab: the size of the input vocabulary
            target_vocab: the size of the target vocabulary
            max_seq_input: the maximum sequence length possible for the input
            max_seq_target: the maximum sequence length possible for the target
            drop_rate: the dropout rate
        Public instance attributes:
            encoder: the encoder layer
            decoder: the decoder layer
            linear: a final Dense layer with target_vocab units
        """
        super(Transformer, self).__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab, max_seq_input,
                               drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab, max_seq_target,
                               drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask, look_ahead_mask,
             decoder_mask):
        """
        Method to make a 'call' for a Transformer network forward pass.
        Transformation from inputs to outputs
        Arguments:
            inputs: tensor of shape (batch, input_seq_len)
                the inputs
            target: tensor of shape (batch, target_seq_len)
                the target
            training: boolean to determine if the model is training
            encoder_mask: the padding mask to be applied to the encoder
            look_ahead_mask: the look ahead mask to be applied to the decoder
            decoder_mask: the padding mask to be applied to the decoder
        Returns:
            a tensor of shape (batch, target_seq_len, target_vocab)
                the transformer output
        """
        enc_output = self.encoder(inputs, training, encoder_mask)
        dec_output = self.decoder(target, enc_output, training,
                                  look_ahead_mask, decoder_mask)
        final_output = self.linear(dec_output)

        return final_output
