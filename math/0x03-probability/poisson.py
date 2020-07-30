#!/usr/bin/env python3
class Poisson:
    """Poisson probability distribution class"""

    def __init__(self, data=None, lambtha=1.):
        """Poisson object attributes initialization
            Args:
                data: A list of the data to be used to estimate
                    the distribution
                lambtha: Expected number of occurences in a given
                    time frame
            Returns:
            Raises:
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
