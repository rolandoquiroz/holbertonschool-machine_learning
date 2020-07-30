#!/usr/bin/env python3
"""Poisson class module"""


class Poisson:
    """Poisson probability distribution class"""

    def __init__(self, data=None, lambtha=1.):
        """Poisson object attributes initialization
            Args:
                data (list): A list of the data to be used to estimate
                    the distribution
                lambtha (float): Expected number of occurences in a given
                    time frame
            Raises:
                TypeError: If data is not a list
                ValueError: If data does not contain at least two data points
                ValueError: If lambtha is not a positive value
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        elif data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data)/len(data))
