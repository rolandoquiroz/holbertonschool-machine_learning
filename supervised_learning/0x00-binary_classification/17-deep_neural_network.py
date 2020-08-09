#!/usr/bin/env python3
"""DeepNeuralNetwork class module"""
import numpy as np


class DeepNeuralNetwork:
    """DeepNeuralNetwork defines a neural network with one hidden layer
        performing binary classification
    """

    def __init__(self, nx, layers):
        """DeepNeuralNetwork object attributes initialization

        Args:
            nx (int): Number of input features to the DeepNeuralNetwork
            layers (int) : List representing the number of nodes in each
                layer of the network

        Attributes:
            L (int): The number of layers in the neural network.
            cache (dict): Intermediary values of the network.
                Shape (layers, 1). It is set to an empty dictionary.
            weights (dict): All weights and biased of the network.
                -The weights of the networkh are initialized using
                the He et al. method and saved in the weights
                dictionary using the key W{l} where {l}
                is the hidden layer the weight belongs to.
                -The biases of the network are initialized to 0’s
                and saved in the weights dictionary using the key
                b{l} where {l} is the hidden layer the bias belongs to.

        Raises:
            TypeError: If nx is not an integer
                       If layers is not a list
                       If the elements in layers are not all positive integers
            ValueError: If nx is less than 1 or if layers is less than 1
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")
        self.nx = nx
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if ((type(layers[i]) is not int) or (layers[i] < 1)):
                raise TypeError("layers must be a list of positive integers")
            self.weights["b{}".format(i+1)] = np.zeros((layers[i], 1))
            if i == 0:
                self.weights["W{}".format(i+1)] = (np.random.randn(layers[i],
                                                   self.nx)*np.sqrt(2/self.nx))
            else:
                self.weights["W{}".format(i+1)] = (np.random.randn(layers[i],
                                                   layers[i-1]) *
                                                   np.sqrt(2/layers[i-1]))

    @property
    def L(self):
        """
        L getter method
        """
        return self.__L

    @property
    def cache(self):
        """
        cache getter method
        """
        return self.__cache

    @property
    def weights(self):
        """
        weights getter method
        """
        return self.__weights
