#!/usr/bin/env python3
"""4-moving_average module
contains the function shuffle_data
"""
import numpy as np


def moving_average(data, beta):
    """Calculates the weighted moving average of a data set.
        Actually exponential moving average is calculated
    Args:
        data: `list` of data to calculate the moving average of.
        beta: `float, the weight used for the moving average.

    Returns:
        `list` containing the moving averages of data.
    """
    exponential_moving_averages = []

    vt = 0
    for i in range(len(data)):
        vt = beta * vt + (1 - beta) * data[i]
        exponential_moving_averages.append(value / (1 - beta**(i + 1)))

    return exponential_moving_averages