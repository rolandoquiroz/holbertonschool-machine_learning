#!/usr/bin/env python3
"""Binomial class module"""


class Binomial:
    """Binomial probability distribution class"""

    e = 2.7182818285
    π = 3.1415926536

    def __init__(self, data=None, n=1, p=0.5):
        """Binomial object attributes initialization
            Args:
                data (list): A list of the data to be used to estimate
                    the distribution
                n (int): The number of Bernoulli trials
                p (float): The probability of a 'success'
            Raises:
                TypeError: If data is not a list
                ValueError: If data does not contain at least two data points
                            If lambtha is not a positive value
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            self.n = round(n)
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.p = float(p)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                for value in data:
                    if type(value) not in [float, int]:
                        raise ValueError("data must contain multiple values")
                mean = sum(data)/len(data)
                variance = sum([(mean-xi)**2 for xi in data])/(len(data)-1)
                p = 1-variance/mean
                self.n = round(mean/p)
                self.p = mean/self.n
