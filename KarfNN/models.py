import numpy as np

from KarfNN import losses
from KarfNN import optimizers
from KarfNN.layer import Dropout


class Karf:
    def __init__(self):
        self.layers = []

    def init(self, X, y):
        self.X = X

        self.y = y
        if len(y.shape) == 1:
            self.y = self.y.reshape(-1, 1)

        # dimension of a sample in the input matrix
        self.input_size = X.shape[1]

    def _initialize_parameters(self):
        prev_dims = self.input_size
        for layer in self.layers:
            # Dropout layer has no parameters to initialize -> skip.
            if isinstance(layer,Dropout):
                continue
            layer.initialize_parameters(prev_dims)
            prev_dims = layer.n_units

    def run(self, n_epochs=100, batch_size=10, learning_rate=0.01, snap_shots=100, optimizer="adam", beta1=0.9,
            beta2=0.999, epsilon=1e-8,loss="mean_squared"):
        assert len(self.layers) != 0, "Please add at least one layer to your Network"

        # initialize parameters
        self._initialize_parameters()
        # call and initialize the chosen optimiser
        optimizer = optimizers.get(optimizer)()
        optimizer.initialize(self.layers)
        self.loss_f = losses.get(loss)
        # initialize mini batchs
        batchs = self._init_batchs(batch_size=batch_size)

        for i in range(n_epochs):
            for batch in batchs:

                batch_X, batch_y = batch
                # Forward propagation
                a = self._forward_prop(batch_X)

                # Compute cost
                cost = np.mean(self.loss_f(a, batch_y))

                # Backward propagation
                self._back_prop(a, batch_y)

                # Update parameters
                optimizer.update_parameters(self.layers, learning_rate, epsilon=epsilon, beta1=beta1, beta2=beta2)

            # Print the cost
            if i % snap_shots == 0:
                print ("Cost after iteration %i: %f" % (i, cost))

    def add(self, layer):
        # add a layer to the network
        self.layers.append(layer)

    def _back_prop(self, a, y):
        # Initializing the backward propagation
        dA_prev = self.loss_f(a, y,prime=True)
        # loop through layers starting from the last one
        for l in reversed(range(len(self.layers))):
            # if it's a dropout layer skip.
            if isinstance(self.layers[l],Dropout):
                continue
            dA_prev, dW_temp, db_temp = self.layers[l].backward(dA_prev)
            # update gradients for the l(th) layer
            self.layers[l].grads["dA"] = dA_prev
            self.layers[l].grads["dW"] = dW_temp
            self.layers[l].grads["db"] = db_temp

    def _forward_prop(self, X):
        # Initializing the forward propagation
        a = X
        for layer in self.layers:
            a = layer.forward(a)
        return a

    # def _update_parameters(self, learning_rate):
    #
    #    for layer in self.layers:
    #        layer.W = layer.W - learning_rate * layer.grads["dW"]
    #        layer.b = layer.b - learning_rate * layer.grads["db"]

    def predict(self, X):
        for layer in self.layers:
            if isinstance(layer,Dropout):
                # don't use the Dropout mask for prediction
                X = layer.forward(X,use_mask=False)
                continue
            X = layer.forward(X)
        return X

    def _init_batchs(self, batch_size):
        # initialize mini batchs
        batchs = []
        m = self.X.shape[0]
        # calculate the number of complete mini batchs
        n_full_batchs = int(m / batch_size)
        # create mini batchs
        for k in range(n_full_batchs):
            batch_X = self.X[k * batch_size:(k + 1) * batch_size, :]
            batch_Y = self.y[k * batch_size:(k + 1) * batch_size, :]

            batchs.append((batch_X, batch_Y))
        # add the last batch which is a smaller mini batch
        if m % batch_size != 0:
            batch_X = self.X[n_full_batchs * batch_size:(n_full_batchs + 1) * batch_size, :]
            batch_Y = self.y[n_full_batchs * batch_size:(n_full_batchs + 1) * batch_size, :]
            batchs.append((batch_X, batch_Y))

        return batchs
