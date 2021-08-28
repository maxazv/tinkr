import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

from dense import Dense
from convolutional import Conv
from reshape import Reshape
from activations import Sigmoid
from loss import binary_cross_entropy, binary_cross_entropy_prime

def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    print(x.shape, y.shape)
    x, y = x[all_indices], y[all_indices]
    print(x.shape, y.shape)
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = np_utils.to_categorical(y)
    y = y.reshape(len(y), 2, 1)
    return x, y

def train_sgd(network, epochs, learning_rate, train_x, train_y):
    for e in range(epochs):
        error = 0
        for x, y in zip(train_x, train_y):
            # forward
            output = x
            for layer in network:
                output = layer.fforward(output)

            error += binary_cross_entropy(y, output)

            # backward
            gradient = binary_cross_entropy_prime(y, output)
            for layer in reversed(network):
                gradient = layer.bprop(gradient, learning_rate, 1)

        error /= len(train_x)
        print(f"{e + 1}/{epochs}, error={error}")

def test_cnn(network, x_t, y_t):
    for x, y in zip(x_t, y_t):
        output = x
        for layer in network:
            output = layer.fforward(output)
        print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")


def main():
    # load MNIST from server, limit to 100 images per class since we're not training on GPU
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = preprocess_data(x_train, y_train, 100)
    x_test, y_test = preprocess_data(x_test, y_test, 100)

    # neural network
    network = [
        Conv((1, 28, 28), 3, 5),
        Sigmoid(),
        Reshape((5, 26, 26), (5 * 26 * 26, 1)),
        Dense(5 * 26 * 26, 100),
        Sigmoid(),
        Dense(100, 2),
        Sigmoid()
    ]
    epochs = 20
    lr = 0.01
    train_sgd(network, epochs, lr, x_train, y_train)
    # test_cnn(network, x_test, y_test)


main()
