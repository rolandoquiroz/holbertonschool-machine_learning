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
        b getter method
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
            using sigmoid activation function

        Parameters
        ----------
        X : numpy.ndarray
            Input data with shape (nx, m)

        Returns
        -------
        self.__A : float
            The forward propagation of the neuron using sigmoid
            activation function
        """
        output = np.matmul(self.__W, X) + self.__b
        self.__A = 1/(1 + np.exp(-output))
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression

        Parameters
        ----------
        Y : numpy.ndarray
            Correct labels for the input data with shape (1, m)
        A : numpy.ndarray
            Activated output of the neuron for each example with shape (1, m)

        Returns
        -------
        cost : float
            The cost of the model using logistic regression
        """
        cost = -np.sum(Y*np.log(A)+(1-Y)*np.log(1.0000001 - A))/Y.shape[1]
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions

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
        prediction = np.where(self.forward_prop(X) >= 0.5, 1, 0)
        cost = self.cost(Y, self.forward_prop(X))
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron
            updating __W and __b

        Parameters
        ----------
        X : numpy.ndarray
            Input data with shape (nx, m)
        Y : numpy.ndarray
            Correct labels for the input data with shape (1, m)
        A : numpy.ndarray
            Activated output of the neuron for each example with shape (1, m)
        alpha : float
            Learning rate
        """
        self.__W = self.__W-alpha*np.matmul(X, (A-Y).T)/Y.shape[1]
        self.__b = self.__b-alpha*np.sum(A-Y)/Y.shape[1]
