#!/usr/bin/env python3
"""Neuron class module"""
import numpy as np


class Neuron:
    """Neuron defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """Neuron object attributes initialization

            Args:
                nx (int): Number of input features to the neuron

            Raises:
                TypeError: If nx is not an integer
                ValueError: If nx is less than 1
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.normal(0, 1, (1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        W getter method
        """
        return self.__W

    @property
    def b(self):
        """
        xb getter method
        """
        return self.__b

    @property
    def A(self):
        """
        A getter method
        """
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron

            Args:
                X (numpy.ndarray): Input data with shape (nx, m)

            Returns: The forward propagation of the neuron
        """
        output = np.matmul(self.__W, X) + self.__b
        self.__A = 1/(1 + np.exp(-output))
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression

            Args:
                Y (numpy.ndarray): Correct labels for the input data
                    with shape (1, m)
                X (numpy.ndarray): Activated output of the neuron for
                    each example with shape (1, m)
            Returns:
                cost (float): The cost of the model using logistic regression
        """
        cost = -np.sum((Y*np.log(A)+(1-Y)*np.log(1.0000001 - A)))/Y.shape[1]
        return cost