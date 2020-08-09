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

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network
            using sigmoid activation function

        Parameters
        ----------
        X : numpy.ndarray
            Input data with shape (nx, m)

        Returns
        -------
        float
            The forward propagation of the neural nework using sigmoid
            activation function in A1 (outputs for the hidden layer)
            and A2 (the output neuron)
        """
        z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))
        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression

        Parameters
        ----------
        Y : numpy.ndarray
            Correct labels for the input data with shape (1, m)
        X : numpy.ndarray
            Activated output of the neuron for each example with shape (1, m)

        Returns
        -------
        cost : float
            The cost of the model using logistic regression
        """
        cost = -np.sum(Y*np.log(A)+(1-Y)*np.log(1.0000001 - A))/Y.shape[1]
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions

        Parameters
        ----------
        X : numpy.ndarray
            Input data with shape (nx, m)
        Y : numpy.ndarray
            Correct labels for the input data with shape (1, m)

        Returns
        -------
        prediction : numpy.ndarray
            Predicted labels for each example with shape (1, m)
        cost : float
            The cost of the model using logistic regression
        """
        (A1, A2) = self.forward_prop(X)
        A2 = np.where(self.__A2 >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A2)
        return (A2, cost)
