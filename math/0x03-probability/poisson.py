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
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                for value in data:
                    if type(value) not in [float, int]:
                        raise ValueError("data must contain multiple values")
                self.lambtha = sum(data)/len(data)

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of 'successes'
            Args:
                k (int): Number of 'successes'
            Returns:
                0: If k is out of range
                PMF (int): PMF value for k
        """
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        else:
            e = 2.7182818285
            k_factorial = 1
            for i in range(1, k+1):
                k_factorial *= i
            return ((self.lambtha**k)*(e**(-self.lambtha)))/k_factorial
