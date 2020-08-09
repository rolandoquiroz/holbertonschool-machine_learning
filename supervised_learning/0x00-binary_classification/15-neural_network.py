#!/usr/bin/env python3
"""NeuralNetwork class module"""
import numpy as np
import matplotlib.pyplot as plt


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
        """Evaluates the neural network’s predictions

        Parameters
        ----------
        X : numpy.ndarray
            Input data with shape (nx, m)
        Y : numpy.ndarray
            Correct labels for the input data with shape (1, m)

        Returns
        -------
        A2 : numpy.ndarray
            Predicted labels for each example with shape (1, m)
        J : float
            The cost of the model using logistic regression
        """
        self.forward_prop(X)
        A2 = np.where(self.__A2 >= 0.5, 1, 0)
        J = self.cost(Y, self.__A2)
        return (A2, J)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network
            updating __W1, __b1, __W2, and __b2

        Parameters
        ----------
        X : numpy.ndarray
            Input data with shape (nx, m)
        Y : numpy.ndarray
            Correct labels for the input data with shape (1, m)
        A1 : numpy.ndarray
            Activated output of the hidden layer for each example with
            shape (1, m)
        A2 : numpy.ndarray
            The predicted output of the neural network for each example with
            shape (1, m)
        alpha : float
            Learning rate
        """
        dZ2 = A2-Y
        m = A1.shape[1]
        dW2 = np.matmul(dZ2, A1.T)/m
        db2 = np.sum(dZ2, axis=1, keepdims=True)/m
        dZ1 = np.matmul(self.__W2.T, dZ2)*(A1*(1-A1))
        dW1 = np.matmul(dZ1, X.T)/m
        db1 = np.sum(dZ1, axis=1, keepdims=True)/m
        self.__W1 = self.__W1-alpha*dW1
        self.__W2 = self.__W2-alpha*dW2
        self.__b1 = self.__b1-alpha*db1
        self.__b2 = self.__b2-alpha*db2

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the neural network

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

        steps = []
        costs = []
        for i in range(iterations + 1):
            (A2, J_i) = self.evaluate(X, Y)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
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

        return (A2, J_i)
