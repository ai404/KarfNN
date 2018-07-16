import numpy as np

from KarfNN.layer import Dropout


class adam:
    def __init__(self, layers=None):
        if layers != None:
            self.initialize(layers)

    def initialize(self, layers):
        # initialize velosity and EWA of squared gradients dictionaries
        for layer in layers:
            # skip Dropout layers
            if isinstance(layer, Dropout):
                continue
            layer.v["dW"] = np.zeros_like(layer.W)
            layer.v["db"] = np.zeros_like(layer.b)
            layer.s["dW"] = np.zeros_like(layer.W)
            layer.s["db"] = np.zeros_like(layer.b)

    def update_parameters(self, layers, learning_rate=0.01, beta1=0.9, beta2=0.99, epsilon=1e-8):

        for layer in layers:
            if isinstance(layer, Dropout):
                continue
            layer.v["dW"] = layer.v["dW"] * beta1 + layer.grads["dW"] * (1 - beta1)
            layer.v["db"] = layer.v["db"] * beta1 + layer.grads["db"] * (1 - beta1)

            layer.v_corrected["dW"] = layer.v["dW"] / (1 - beta1 ** 2)
            layer.v_corrected["db"] = layer.v["db"] / (1 - beta1 ** 2)

            layer.s["dW"] = layer.s["dW"] * beta2 + (1 - beta2) * layer.grads["dW"] ** 2
            layer.s["db"] = layer.s["db"] * beta2 + (1 - beta2) * layer.grads["db"] ** 2

            layer.s_corrected["dW"] = layer.s["dW"] / (1 - beta2 ** 2)
            layer.s_corrected["db"] = layer.s["db"] / (1 - beta2 ** 2)

            # update weights and bias
            layer.W -= learning_rate * layer.v_corrected["dW"] / (np.sqrt(layer.s_corrected["dW"]) + epsilon)
            layer.b -= learning_rate * layer.v_corrected["db"] / (np.sqrt(layer.s_corrected["db"]) + epsilon)


class momentum:
    def __init__(self, layers=None):
        if layers is not None:
            self.initialize(layers)

    def initialize(self, layers):
        # initialize velosity dictionary for each Layer
        for layer in layers:
            if isinstance(layer, Dropout):
                continue
            layer.v["dW"] = np.zeros_like(layer.W)
            layer.v["db"] = np.zeros_like(layer.b)

    def update_parameters(self, layers, learning_rate=0.01, beta1=0.9, *args, **kwargs):
        for layer in layers:
            if isinstance(layer, Dropout):
                continue
            layer.v["dW"] = layer.v["dW"] * beta1 + layer.grads["dW"] * (1 - beta1)
            layer.v["db"] = layer.v["db"] * beta1 + layer.grads["db"] * (1 - beta1)

            # update weights and bias
            layer.W -= learning_rate * layer.v["dW"]
            layer.b -= learning_rate * layer.v["db"]


optimizers = globals()


# Get Activation Function
def get(name):
    assert name in optimizers
    return adam if name is None else optimizers[name]
