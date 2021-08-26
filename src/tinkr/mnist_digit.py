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
        for i in range(X.shape[1]):
            # feedforward
            output = X[:, i, None]
            for layer in tinkr:
                output = layer.fforward(output)

            error += mse(Y[:, i, None], output)
            if i == 0:
                print("First Epoch Err: ", error)

            # backpropagation
            local_gradient = mse_prime(Y[:, i, None], output)
            for layer in reversed(tinkr):
                local_gradient = layer.bprop(local_gradient, lr)

        error /= X.shape[1]
        print(f"{e + 1}/{epochs}, error={error}")
        if error < 0.025:
            return tinkr
    return tinkr

def predict(X, tinkr):
    print()
    z = X
    for layer in tinkr:
        z = layer.fforward(z)
    print(np.argmax(z))
    #print(np.argmax(z))
def predict_sgd(X, Y, tinkr):
    print()
    for i in range(X.shape[1]):
        z = X[:, i, None]
        for layer in tinkr:
            z = layer.fforward(z)
        print(np.argmax(Y[:, i, None]), "->", np.around(z.reshape(1, z.size), 2))
        print("Err: ", mse(Y[:, i, None], z))

def save_model(model, path='../../res/models.npz'):
    copy = []
    for i in range(len(model)):
        if i % 2 == 0:
            copy.append(model[i].w)
            copy.append(model[i].b)
    np.savez(path, *copy)
def load_model(ai, path='../../res/trained/uh.npz'):
    model = np.load(path)
    key = 'arr_'
    c = 0
    for i in range(len(ai)):
        if i % 2 == 0:
            ai[i].w = model[key + str(c*2)]
            ai[i].b = model[key + str((c*2)+1)]
            c += 1
    return ai

def main():
    # data/ hyperparameter configuration
    # minibatches, tinkr, _, _ = config_data([784, 20, 20, 10], 10)
    digit_rec = [Dense(784, 20),
                 Tanh(),
                 Dense(20, 20),
                 Tanh(),
                 Dense(20, 10),
                 ReLU()]
    _, _, X, Y = config_data([], 1)
    x_sgd = X
    y_sgd = DataConfig.one_hot(Y)
    epochs = 7
    lr = 0.008  # 0.008 -> error=0.002290681273446552

    digit_rec = train_stochastic(x_sgd, y_sgd, epochs, lr, digit_rec)
    predict_sgd(x_sgd[:, 0:10], y_sgd[:, 0:10], digit_rec)
    save_model(digit_rec)

    # tinkr = train_minibatch(epochs, minibatches, lr, tinkr, 10)
    # img = minibatches[0][1][:, 4, None]
    # print(img.shape)
    # predict(img, tinkr)
    # print('Expected: ', minibatches[0][0][4])


main()
