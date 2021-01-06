#!/usr/bin/env python3
"""function positional_encoding"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Function that calculates the positional encoding for a transformer.

    Arguments:
        max_seq_len: int
            the maximum sequence length
        dm:
            the model depth

    Returns:
        positional_encoding_vectors: numpy.ndarray of shape (max_seq_len, dm)
            positional encoding vectors
    """
    pe = np.zeros([max_seq_len, dm])
    for pos in range(max_seq_len):
        for i in range(0, dm, 2):
            pe[pos, i] = np.sin(pos / (10000 ** ((2 * i)/dm)))
            pe[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1))/dm)))
    return pe
