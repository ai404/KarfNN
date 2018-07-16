import numpy as np


# linear activation
def linear(x, prime=False):
    if prime:
        return 1
    return x


# sigmoid activation
def sigmoid(x, prime=False):
    if prime:
        return sigmoid(x) / (1 + np.exp(-x))
    return 1 / (1 + np.exp(-x))


# relu activation
def relu(x, prime=False):
    if prime:
        return x > 0
    return np.maximum(0, x)


# tanh activation
def tanh(x, prime=False):
    if prime:
        return 1 / (np.cosh(x) ** 2)
    return np.tanh(x)


# softmax activation
def softmax(x, prime=False):
    if prime:
        return x / x.shape[0]
    exp = np.exp(x - np.max(x))
    return exp / np.sum(exp, axis=1, keepdims=True)


activations = globals()


# Get Activation Function
def get(name):
    assert name in activations
    return linear if name is None else activations[name]
