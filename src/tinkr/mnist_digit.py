import random

from dense import Dense
from activations import ReLU
from activations import Tanh
from loss import mse, mse_prime
from data_config import DataConfig

import numpy as np


def config_data(layer_shapes, batch_size):
    dc = DataConfig()
    dc.load('../../res/train.csv')
    x_t, y_t = dc.setup_data()
    x_t = dc.normalize(x_t, 255.)
    # print(x_t[:, from: to])
    minibatch_digits = dc.create_batches(x_t, y_t, batch_size)

    tinkr = []
    for i in range(len(layer_shapes)-1):
        tinkr.append(Dense(layer_shapes[i], layer_shapes[i+1]))
        tinkr.append(ReLU())

    return minibatch_digits, tinkr, x_t, y_t
def train_minibatch(epochs, mb, lr, tinkr, batch_size):
    for e in range(epochs):
        error = 0
        for batch in mb:
            # feedforward
            onehot_y = DataConfig.one_hot(batch[0])
            output = batch[1]
            for layer in tinkr:
                output = layer.fforward(output)
            error += mse(onehot_y, output)

            # backpropagation
            gradient = mse_prime(onehot_y, output)
            for layer in reversed(tinkr):
                gradient = layer.bprop(gradient, lr, batch_size)

        error /= len(mb)
        print(f"{e + 1}/{epochs}, error={error}")
        random.shuffle(mb)
    return tinkr
def train_stochastic(X, Y, epochs, lr, tinkr):
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
                local_gradient = layer.bprop(local_gradient, lr)

        error /= len(X)
        print(f"{e + 1}/{epochs}, error={error}")
    return tinkr
def predict(X, tinkr):
    print()
    z = X
    for layer in tinkr:
        z = layer.fforward(z)
    print(np.argmax(z))
    #print(np.argmax(z))

def main():
    # data/ hyperparameter configuration
    # minibatches, tinkr, _, _ = config_data([784, 20, 20, 10], 10)
    digit_rec = [Dense(784, 15),
                 ReLU(),
                 Dense(15, 15),
                 Tanh(),
                 Dense(15, 10),
                 ReLU()]
    _, _, X, Y = config_data([], 1)
    x_sgd = X.reshape(X.shape[1], X.shape[0], 1)
    y_sgd = DataConfig.one_hot(Y)
    y_sgd = y_sgd.reshape(y_sgd.shape[1], y_sgd.shape[0], 1)
    print(x_sgd.shape, y_sgd.shape)
    epochs = 200
    lr = 0.02
    trained = train_stochastic(x_sgd, y_sgd, epochs, lr, digit_rec)
    predict(x_sgd, trained)

    # tinkr = train_minibatch(epochs, minibatches, lr, tinkr, 10)
    # img = minibatches[0][1][:, 4, None]
    # print(img.shape)
    # predict(img, tinkr)
    # print('Expected: ', minibatches[0][0][4])


main()
