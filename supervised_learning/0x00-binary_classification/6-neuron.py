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
        A : numpy.ndarray
            The forward propagation of the neuron using sigmoid
            activation function. Y_hat
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1/(1 + np.exp(-Z))
        return self.__A

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
        J : float
            The cost of the model using logistic regression
        """
        J = -np.sum(Y*np.log(A)+(1-Y)*np.log(1.0000001 - A))/Y.shape[1]
        return J

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
        A : numpy.ndarray
            Predicted labels for each example with shape (1, m)
        J : float
            The cost of the model using logistic regression
        """
        self.forward_prop(X)
        A = np.where(self.__A >= 0.5, 1, 0)
        J = self.cost(Y, self.__A)
        return A, J

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
        dZ = A-Y
        m = Y.shape[1]
        dW = np.matmul(dZ, X.T)/m
        db = np.sum(dZ)/m
        self.__W = self.__W-alpha*dW
        self.__b = self.__b-alpha*db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the neuron

        Parameters
        ----------
        X : numpy.ndarray
            Input data with shape (nx, m)
        Y : numpy.ndarray
            Correct labels for the input data with shape (1, m)
        iterations : int
            Number of iterations to train over
        alpha : float
            Learning rate

        Returns
        -------
        A : numpy.ndarray
            Predicted labels for each example with shape (1, m)
        J : float
            The cost of the model using logistic regression

        Raises
        ------
        TypeError
            If `iterations` is not an integer
            If `alpha` is not a float
        ValueError
            If `iterations` is not positive
            If `alpha` is not positive
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for _ in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
        return self.evaluate(X, Y)
