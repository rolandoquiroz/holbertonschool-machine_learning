#!/usr/bin/env python3
"""function from_file"""
import pandas as pd


def from_file(filename, delimiter):
    """
    Function that loads data from a file as a pd.DataFrame.

    Parameters
    ----------
    filename: str
        file to load from
    delimiter : str
        the column separator

    Returns
    -------
    df: Pandas Dahaframe
        the loaded pd.DataFrame
    """
    df = pd.read_csv(filename, sep=delimiter)
    return df
