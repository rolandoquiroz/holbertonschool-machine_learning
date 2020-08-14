#!/usr/bin/env python3
"""DeepNeuralNetwork class module"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """DeepNeuralNetwork defines a neural network with one hidden layer
        performing binary classification
    """
    def __init__(self, nx, layers, activation='sig'):
        """DeepNeuralNetwork object attributes initialization
        Args:
            nx (int): Number of input features to the DeepNeuralNetwork
            layers (int): List representing the number of nodes in each
                layer of the network
            activation (str): Represents the type of activation function
                used in the hidden layers
                    - sig represents a sigmoid activation
                    - tanh represents a tanh activation
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
        if activation is not "sig" or activation is not "tanh":
            raise ValueError("activation must be 'sig' or 'tanh'")
        self.__activation = activation

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

    @property
    def activation(self):
        """
        activation getter method
        """
        return self.__activation

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network
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
        self.__cache['A0'] = X
        for i in range(self.__L):
            Z = (np.matmul(self.__weights["W"+str(i+1)],
                 self.__cache["A"+str(i)]) +
                 self.__weights["b"+str(i+1)])
            if i != self.__L - 1:
                if self.__activation == 'tanh':
                    self.__cache["A"+str(i+1)] = np.tanh(Z)
                if self.__activation == 'sig':
                    self.__cache["A"+str(i+1)] = 1/(1+np.exp(-Z))
            else:
                temp = np.exp(Z)
                self.__cache["A"+str(i+1)] = (temp/np.sum(temp, axis=0,
                                                          keepdims=True))
        return self.__cache["A"+str(i+1)], self.__cache

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
        J = (-1/(Y.shape[1]))*np.sum(Y*np.log(A))
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
        aux = np.amax(self.__cache["A"+str(self.__L)], axis=0)
        return (np.where(self.__cache["A"+str(self.__L)] == aux, 1, 0),
                self.cost(Y, self.__cache["A"+str(self.__L)]))

    def gradient_descent(self, Y, cache, alpha=0.05):
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
        A2 = cache["A{}".format(self.__L)]
        dA2 = A2-Y
        m = Y.shape[1]
        for i in range(self.__L, 0, -1):
            A2 = cache["A"+str(i)]
            if self.__activation == 'tanh':
                g_dot = 1-A2**2
            if self.__activation == 'sig':
                g_dot = A2*(1-A2)
            if i == self.__L:
                dZ = dA2
            if i != self.__L:
                dZ = dA2*g_dot
            A1 = cache["A"+str(i-1)]
            dW = np.matmul(dZ, A1.T)/m
            db = np.sum(dZ, axis=1, keepdims=True)/m
            dA2 = np.matmul((self.__weights["W"+str(i)]).T, dZ)
            self.__weights["W"+str(i)] = self.__weights["W"+str(i)]-alpha*dW
            self.__weights["b"+str(i)] = self.__weights["b"+str(i)]-alpha*db

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
        A2 : numpy.ndarray
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
        for i in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, self.cache, alpha)
            if i % step == 0:
                if graph is True:
                    steps.append(i)
                    costs.append(self.cost(Y,
                                 self.cache["A{}".format(self.L)]))
                if verbose is True:
                    print("Cost after {} iterations: {}"
                          .format(i, self.cost(Y, A)))
        J = self.evaluate(X, Y)[1]
        if verbose is True:
            print("Cost after {} iterations: {}".format(i + 1, J))
        if graph is True:
            plt.plot(steps, costs)
            plt.title('Training Cost')
            plt.xlabel('iteration')
            plt.ylabel('cost')
        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file in pickle format
        Args:
        filename : The file to which the object should be saved
            If filename does not have the extension .pkl, add it
        """
        if filename is not None:
            if not filename.endswith('.pkl'):
                filename = filename + '.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object
        Args:
        filename : Is the file from which the object should be loaded
        Returns: the loaded object, or None if filename doesn’t exist
        """
        with open(filename, 'rb') as file:
            file_ready_to_go = pickle.load(file)
            return (file_ready_to_go)
