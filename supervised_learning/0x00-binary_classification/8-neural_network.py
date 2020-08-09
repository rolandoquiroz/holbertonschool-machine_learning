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
        self.W1 = np.random.normal(0, 1, (nodes, nx))
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.normal(0, 1, (1, nodes))
        self.b2 = 0
        self.A2 = 0
