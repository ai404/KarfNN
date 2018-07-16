import numpy as np


# Mean Squared
def mean_squared(a, y, prime=False):
    if prime:
        return a - y
    return np.mean(np.square(a - y))


# Cross Entropy
def cross_entropy(a, y, prime=False):
    if prime:
        dA = np.copy(a)
        dA[range(a.shape[0]), np.argmax(y, axis=1)] -= 1
        return dA
    return -np.log(a[range(a.shape[0]), np.argmax(y, axis=1)])


losses = globals()


# Get Loss Function
def get(name):
    assert name in losses,"loss function can either be mean_squared or cross_entropy"
    return losses[name]
