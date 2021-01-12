#!/usr/bin/env python3
"""Create Masks"""
import tensorflow.compat.v2 as tf


def create_padding_mask(seq):
    """
    Mask all the pad tokens in the batch of sequence.
    It ensures that the model does not treat padding as the input.
    The mask indicates where pad value 0 is present:
    it outputs a 1 at those locations, and a 0 otherwise.
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    """
    The look-ahead mask is used to mask the future tokens in a sequence.
    In other words, the mask indicates which entries should not be used.

    This means that to predict the third word, only the first and second
    word will be used. Similarly to predict the fourth word, only the first,
    second and the third word will be used and so on.
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_masks(inputs, target):
    """
    Function that creates all masks for training/validation

    Arguments:
        inputs: tf.Tensor of shape (batch_size, seq_len_in)
            the input sentence
        target: tf.Tensor of shape (batch_size, seq_len_out)
            the target sentence

    Returns:
        encoder_mask, look_ahead_mask, decoder_mask
        encoder_mask: tf.Tensor padding mask of shape
            (batch_size, 1, 1, seq_len_in) to be applied in the encoder
        look_ahead_mask: tf.Tensor look ahead mask of shape
            (batch_size, 1, seq_len_out, seq_len_out)
            to be applied in the decoder
        decoder_mask: tf.Tensor padding mask of shape
            (batch_size, 1, 1, seq_len_in)
            to be applied in the decoder
    """
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inputs)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    dec_padding_mask = create_padding_mask(inputs)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    seq_len_out = target.shape[1]
    look_ahead_mask = create_look_ahead_mask(seq_len_out)
    dec_target_padding_mask = create_padding_mask(target)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask
