#!/usr/bin/env python3
"""Neuron class module"""
import numpy as np


class Neuron:
    """Neuron defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """Neuron object attributes initialization

        Args:
            nx (int): Number of input features to the neuron

        Attributes:
            W (numpy.ndarray): Weights vector for the neuron.
                Shape (1, nx). It is initialized using a random
                normal distribution.
            b (float): The bias for the neuron.
                It is initialized to 0.
            A (float): The activated output of the neuron (prediction).
                It is initialized to 0.

        Raises:
            TypeError: If nx is not an integer
            ValueError: If nx is less than 1
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.normal(0, 1, (1, nx))
        self.b = 0
        self.A = 0
