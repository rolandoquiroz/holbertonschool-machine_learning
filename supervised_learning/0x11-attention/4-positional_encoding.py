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
        PE: numpy.ndarray of shape (max_seq_len, dm)
            positional encoding vectors
    """
    PE = np.zeros([max_seq_len, dm])
    for pos in range(max_seq_len):
        for i in range(0, dm, 2):
            PE[pos, i] = np.sin(pos / (10000 ** (2 * (i // 2) / dm)))
            PE[pos, i + 1] = np.cos(pos / (10000 ** (2 * (i // 2) / dm)))
    return PE
