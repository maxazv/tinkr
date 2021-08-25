import random

from dense import Dense
from activations import ReLU
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

    return minibatch_digits, tinkr
def train(epochs, mb, lr, tinkr, batch_size):
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
def predict(X, tinkr):
    print()
    z = X
    for layer in tinkr:
        z = layer.fforward(z)

def main():
    # data/ hyperparameter configuration
    minibatches, tinkr = config_data([784, 20, 20, 10], 10)
    # plot data
    # DataConfig.plot_data(minibatches[0][1][:, 9])
    # print("Plotted: ", minibatches[0][0][9])
    # print("Plotted Onehot: \n", DataConfig.one_hot(minibatches[0][0]))
    # print(minibatches[0][1].shape)
    epochs = 75
    lr = 0.25

    train(epochs, minibatches, lr, tinkr, 10)

    predict(minibatches[0][1], tinkr)
    """
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    c = np.stack((a, b), 1)
    d = np.sum(c, 1)
    d = np.array([d])
    # print(d.reshape((d.size, 1))) """


main()
