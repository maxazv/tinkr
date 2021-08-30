import numpy as np
from layer import Layer

class Dense(Layer):
    def __init__(self, x_n, y_n, momentum=False, beta=0.9):
        """Single Neural Network Layer of shape (`output_n`, `data_n`)"""
        self.w = np.random.rand(y_n, x_n) - 0.5
        self.b = np.random.rand(y_n, 1) - 0.5

        self.momentum = momentum
        self.beta = beta
        self.velB = self.velW = 0

    def fforward(self, x):
        self.input = x
        return self.w.dot(x) + self.b

    def bprop(self, gradient, lr, batch_size=1):
        if self.momentum:
            dB = self.beta*self.velB + (1-self.beta)*(1/batch_size * np.sum(gradient))
            dW = self.beta*self.velW + (1-self.beta)*(1/batch_size * gradient.dot(self.input.T))
            self.velB, self.velW = dB, dW
        else:
            dB = 1/batch_size * np.sum(gradient)
            dW = 1/batch_size * gradient.dot(self.input.T)

        self.b = self.b - lr * dB    # for minibatch sum up over axis 1
        self.w = self.w - lr * dW  # for minibatch stays the same
        return self.w.T.dot(gradient)
