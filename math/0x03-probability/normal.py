#!/usr/bin/env python3
"""Normal class module"""


class Normal:
    """Normal probability distribution class"""

    e = 2.7182818285
    π = 3.1415926536

    def __init__(self, data=None, mean=0., stddev=1.):
        """Normal object attributes initialization
            Args:
                data (list): A list of the data to be used to estimate
                    the distribution
                mean (float): The mean of the distribution
                stddev (float): The standard deviation of the distribution
            Raises:
                TypeError: If data is not a list
                ValueError: If data does not contain at least two data points
                            If lambtha is not a positive value
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                for value in data:
                    if type(value) not in [float, int]:
                        raise ValueError("data must contain multiple values")
                self.mean = sum(data)/len(data)
                self.stddev = (sum((x - self.mean)**2 for x in data) /
                               len(data))**0.5
