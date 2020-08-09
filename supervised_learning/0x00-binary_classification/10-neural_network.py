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
            W1 (numpy.ndarray): Weights vector for the hidden layer.
                Shape (nodes, nx). It is initialized using a random
                normal distribution.
            b1 (numpy.ndarray): The bias for the hidden layer.
                Shape (nodes, 1). It is initialized with 0’s.
            A1 (float): The activated output for the hidden layer.
                It is initialized to 0.
            W2 (numpy.ndarray): The weights vector for the output neuron.
                Shape (1, nodes). It is initialized using a random normal
                distribution.
            b2 (float): The bias for the output neuron.
                It is initialized to 0.
            A2 (float): The activated output for the output neuron
                (prediction). It is initialized to 0.

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

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network
            using sigmoid activation function

        Parameters
        ----------
        X : numpy.ndarray
            Input data with shape (nx, m)

        Returns
        -------
        (A1, A2) : tuple
            The forward propagation of the neural nework using sigmoid
            activation function in A1 (outputs for the hidden layer)
            and A2 (the output neuron)
        """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return (self.__A1, self.__A2)
