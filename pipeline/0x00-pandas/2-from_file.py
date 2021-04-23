#!/usr/bin/env python3
""" Module that contains function: def from_file(filename, delimiter) """
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
    df: Pandas Dathaframe
        the loaded pd.DataFrame
    """
    df = pd.read_csv(filepath_or_buffer=filename, delimiter=delimiter)
    return df
