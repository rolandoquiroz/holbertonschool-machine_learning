#!/usr/bin/env python3
"""function from_numpy"""
import numpy as np
import pandas as pd


def from_numpy(array):
    """
    Function that creates a pd.DataFrame from a np.ndarray.

    Parameters
    ----------
    array : np.ndarray
        the np.ndarray to create the pd.DataFrame

    Returns
    -------
    df = pd.DataFrame
        the newly created pd.DataFrame
    """
    labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    columns = labels[:array.shape[1]]
    return pd.DataFrame(data=array, columns=columns)
