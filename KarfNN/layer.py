import numpy as np

from KarfNN import activations


class Layer:
    def initialize_parameters(self, n):
        pass

    def forward(self, a):
        pass

    def backward(self, da):
        pass


class Dense(Layer):
    def __init__(self, n_units, activation="linear"):
        self.n_units = n_units
        self.activation_f = activations.get(activation)

        # dictionary for gradients for bias and weights
        self.grads = {}
        # dictionary for velocity
        self.v = {}
        self.v_corrected = {}

        # dictionary for exponentially weighted average of the squared gradient
        self.s = {}
        self.s_corrected = {}

    def initialize_parameters(self, prev_n_units):
        # initialize weights with small values
        self.W = np.random.randn(prev_n_units, self.n_units) * 0.1
        # initialize bias vector with zeros
        self.b = np.zeros((1, self.n_units))

    def forward(self, prev_a):
        self.a_linear = prev_a
        self.Z = self._linear_forward()

        return self.activation_f(self.Z)

    def _linear_forward(self):
        # forward propagation for the linear part
        return np.dot(self.a_linear, self.W) + self.b

    def backward(self, dA):
        dZ = np.multiply(dA, self.activation_f(self.Z, prime=True))
        dA_prev, dW, db = self._linear_backward(dZ)

        return dA_prev, dW, db

    def _linear_backward(self, dZ):
        m = self.a_linear.shape[1]

        dW = np.dot(self.a_linear.T, dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m

        dA_prev = np.dot(dZ, self.W.T)

        return dA_prev, dW, db

# we can either call the Class above by fully_connected or a Dense layer
fully_connected = Dense


class Dropout(Layer):
    def __init__(self, threshold=0.8, seed=1):
        assert threshold > 0, "threshhold should be greater than zero and less than or equal 1"
        # specify threshold to filter some units and reset them to 0
        self.threshold = threshold
        # the seed used to generate random values
        self.seed = seed

    def forward(self, X, use_mask=True):
        # set the seed
        np.random.seed(self.seed)
        # generate the mask matrix
        self.mask = (np.random.rand(*X.shape) < self.threshold) / self.threshold
        # return the input matrix with some units set to zero
        return X * self.mask if use_mask else X

    def backward(self, do, use_mask=True):
        # backward propagation with mask in case of training and without mask in case of predicting
        return do * self.mask if use_mask else do
