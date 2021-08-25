from dense import Dense
from activations import Tanh
from loss import mse, mse_prime
# from data_config import DataConfig

import numpy as np

def config_data(layer_shapes):
    X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
    Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

    tinkr = []
    for i in range(len(layer_shapes)-1):
        tinkr.append(Dense(layer_shapes[i], layer_shapes[i+1]))
        tinkr.append(Tanh())

    return X, Y, tinkr
def train(epochs, X, Y, lr, tinkr, batch_size=1):
    for e in range(epochs):
        error = 0
        for x, y in zip(X, Y):
            # feedforward
            output = x
            for layer in tinkr:
                output = layer.fforward(output)

            error += mse(y, output)

            # backpropagation
            local_gradient = mse_prime(y, output)
            for layer in reversed(tinkr):
                local_gradient = layer.bprop(local_gradient, lr, batch_size)

        error /= len(X)
        print(f"{e + 1}/{epochs}, error={error}")
def predict(X, tinkr):
    print()
    for x in X:
        z = x
        for layer in tinkr:
            z = layer.fforward(z)
        print(x.tolist(), "->", z)

def main():
    # data/ hyperparameter configuration
    X, Y, tinkr = config_data([2, 3, 1])
    # mb = DataConfig.create_batches(Y, X, 3)
    # minibatch[batch][0 = Y-Values & 1 = X-Values]
    # print(mb[1][1])
    epochs = 3000
    lr = 0.02
    # train the model
    train(epochs, X, Y, lr, tinkr, batch_size=1)
    # predictions by trained model
    predict(X, tinkr)
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    c = np.stack((a, b), 1)
    d = np.sum(c, 1)
    d = np.array([d])
    print(d.reshape((d.size, 1)))


main()
