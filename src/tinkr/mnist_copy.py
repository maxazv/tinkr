import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt

from dense import Dense
from convolutional import Conv
from reshape import Reshape
from activations import Sigmoid
# from loss import binary_cross_entropy, binary_cross_entropy_prime
from loss import mse, mse_prime

def preprocess_data(x, y, limit):
    #zero_index = np.where(y == 0)[0][:limit]
    # one_index = np.where(y == 1)[0][:limit]
    # all_indices = np.hstack((zero_index, one_index))
    # all_indices = np.random.permutation(all_indices)
    all_index = y[:limit]
    all_indices = np.random.permutation(all_index)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = np_utils.to_categorical(y)
    # y = y.reshape(len(y), 2, 1)
    y = y.reshape(len(y), 10, 1)
    return x, y

def train_sgd(network, epochs, learning_rate, train_x, train_y):
    for e in range(epochs):
        error = 0
        for x, y in zip(train_x, train_y):
            output = x
            for layer in network:
                output = layer.fforward(output)
            error += mse(y, output)

            gradient = mse_prime(y, output)
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
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = preprocess_data(x_train, y_train, 60000)
    x_test, y_test = preprocess_data(x_test, y_test, 30)

    network = [
        Conv((1, 28, 28), 3, 5),
        Sigmoid(),
        Reshape((5, 26, 26), (5 * 26 * 26, 1)),  # output_width/height - kernel_size + 1
        Dense(5 * 26 * 26, 100),                 # TODO: Implement MAX POOLING
        Sigmoid(),
        Dense(100, 10),
        Sigmoid()
    ]
    epochs = 10
    lr = 0.005

    #plt.imshow(x_test[0][0], interpolation='none', cmap='gray')
    #plt.show()
    train_sgd(network, epochs, lr, x_train, y_train)
    test_cnn(network, x_test, y_test)


main()
