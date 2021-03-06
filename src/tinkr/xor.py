from dense import Dense
from activations import Tanh
from loss import mse, mse_prime
import numpy as np

def unison_shuffle(a, b, axis):
    copA = np.zeros(a.shape)
    copB = np.zeros(b.shape)
    idx_shuffle = np.arange(0, a.shape[axis])
    np.random.shuffle(idx_shuffle)
    if axis == 1:
        for i in range(len(idx_shuffle)):
            copA[:, idx_shuffle[i]] = a[:,i]
            copB[:, idx_shuffle[i]] = b[:,i]
        return copA, copB

    for i in range(len(idx_shuffle)):
        copA[idx_shuffle[i],:] = a[i,:]
        copB[idx_shuffle[i],:] = b[i,:]
    return copA, copB

def config_data(layer_shapes, moment):
    X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
    Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

    tinkr = []
    for i in range(len(layer_shapes)-1):
        tinkr.append(Dense(layer_shapes[i], layer_shapes[i+1], momentum=moment))
        tinkr.append(Tanh())

    return X, Y, tinkr
def train(epochs, X, Y, lr, tinkr, batch_size=1):
    # stochastic gradient descent (update after each example)
    for e in range(epochs):
        print(X.shape, Y.shape)
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
    X, Y, tinkr = config_data([2, 3, 1], True)
    epochs = 2000   # 2000
    lr = 0.02
    # train the model
    train(epochs, X, Y, lr, tinkr, batch_size=1)
    # predictions by trained model
    predict(X, tinkr)


main()
