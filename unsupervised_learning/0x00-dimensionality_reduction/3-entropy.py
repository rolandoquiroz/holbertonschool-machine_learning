#!/usr/bin/env python3
"""
module 3-entropy
contains the HP function
"""
import numpy as np


def HP(Di, beta):
    """
    Calculates the Shannon entropy and P affinities relative to a data point
    """
    Pi = (np.exp(-Di * beta)) / (np.sum(np.exp(-Di * beta)))
    Hi = -np.sum(Pi * np.log2(Pi))
    return Hi, Pi
