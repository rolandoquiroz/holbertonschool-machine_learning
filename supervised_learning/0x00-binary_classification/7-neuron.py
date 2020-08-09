#!/usr/bin/env python3
"""Neuron class module"""
import numpy as np
import matplotlib.pyplot as plt


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
        J = self.cost(Y, self.forward_prop(X))
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

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the neuron by updating the private attributes
            __W, __b, and __A

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
        verbose : bool
            Defines whether or not to print information about the training
        graph : bool
            Defines whether or not to graph information about the training
            once the training has completed

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
            If either `verbose` or `graph` are True
                and if step is not an integer
        ValueError
            If `iterations` is not positive
            If `alpha` is not positive
            If either `verbose` or `graph` are True
                and if step is not positive or is greater than iterations
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        steps = []
        costs = []
        for i in range(iterations + 1):
            (A, J_i) = self.evaluate(X, Y)
            self.gradient_descent(X, Y, self.__A, alpha)
            if i % step == 0:
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, J_i))
                if graph is True:
                    steps.append(i)
                    costs.append(J_i)

        if graph is True:
            plt.plot(steps, costs)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return (A, J_i)
