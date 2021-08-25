from dense import Dense
from activations import ReLU
from loss import mse, mse_prime
from data_config import DataConfig

# import numpy as np


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
        b_idx = 0
        for batch in mb:
            # feedforward
            onehot_y = DataConfig.one_hot(batch[0])
            output = batch[1]
            for layer in tinkr:
                output = layer.fforward(output)
            error += mse(onehot_y, output)
            """
            print(np.around(output, 3))
            print()
            print(np.sum(output, 1, keepdims=True))
            print()
            print(onehot_y)
            print(error)"""

            # backpropagation
            local_gradient = mse_prime(onehot_y, output)
            for layer in reversed(tinkr):
                local_gradient = layer.bprop(local_gradient, lr, batch_size)

            if b_idx % 100 == 0:
                error /= batch_size
                print(f"{e+1}/{epochs}, {b_idx+1}/{len(mb)}, error={error}")
            b_idx += 1

def predict(X, tinkr):
    print()
    z = X
    for layer in tinkr:
        z = layer.fforward(z)
    # print(X.tolist(), "->", z)

def main():
    # data/ hyperparameter configuration
    minibatches, tinkr = config_data([784, 20, 20, 10], 10)
    # plot data
    # DataConfig.plot_data(minibatches[0][1][:, 9])
    # print("Plotted: ", minibatches[0][0][9])
    # print("Plotted Onehot: \n", DataConfig.one_hot(minibatches[0][0]))
    # print(minibatches[0][1].shape)
    epochs = 100
    lr = 0.01

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
