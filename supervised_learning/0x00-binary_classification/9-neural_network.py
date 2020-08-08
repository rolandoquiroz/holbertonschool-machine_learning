#!/usr/bin/env python3
"""NeuralNetwork class module"""
import numpy as np


class NeuralNetwork:
    """NeuralNetwork defines a neural network with one hidden layer
        performing binary classification
    """

    def __init__(self, nx, nodes):
        """NeuralNetwork object attributes initialization

        Args:
            nx (int): Number of input features to the NeuralNetwork
            nodes (int) : Number of nodes found in the hidden layer

        Attributes:
            W1: Weights vector for the hidden layer. Upon instantiation,
                it should be initialized using a random normal distribution.
            b1: The bias for the hidden layer. Upon instantiation,
                it should be initialized with 0’s.
            A1: The activated output for the hidden layer.
                Upon instantiation, it should be initialized to 0.
            W2: The weights vector for the output neuron. Upon instantiation,
                it should be initialized using a random normal distribution.
            b2: The bias for the output neuron. Upon instantiation,
                it should be initialized to 0.
            A2; The activated output for the output neuron (prediction).
                Upon instantiation, it should be initialized to 0.

        Raises:
            TypeError: If nx is not an integer or if nodes is not an integer
            ValueError: If nx is less than 1 or if nodes is less than 1
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.normal(0, 1, (nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(0, 1, (1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """
        W1 getter method
        """
        return self.__W1

    @property
    def b1(self):
        """
        b1 getter method
        """
        return self.__b1

    @property
    def A1(self):
        """
        A getter method
        """
        return self.__A1

    @property
    def W2(self):
        """
        W getter method
        """
        return self.__W2

    @property
    def b2(self):
        """
        b getter method
        """
        return self.__b2

    @property
    def A2(self):
        """
        A getter method
        """
        return self.__A2
