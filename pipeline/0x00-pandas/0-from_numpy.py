#!/usr/bin/env python3
''' Module that contains function: def from_numpy(array) '''
import numpy as np
import pandas as pd


def from_numpy(array):
    '''
    Function that creates a pd.DataFrame from a np.ndarray

    Parameters
    ----------
    array : np.ndarray
        array is the np.ndarray from which you create the pd.DataFrame

    Returns
    -------
    df : pd.DataFrame
        the newly created pd.DataFrame

    Note
    ----
    The columns of the pd.DataFrame should be labeled in alphabetical order
    and capitalized. There will not be more than 26 columns.
    '''
    labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    columns = labels[0:np.shape(array)[1]]
    df = pd.DataFrame(data=array, columns=columns)
    return df
