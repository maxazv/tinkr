import numpy as np
from layer import Layer

class Dense(Layer):
    def __init__(self, x_n, y_n):
        """Single Neural Network Layer of shape (`output_n`, `data_n`)"""
        self.w = np.random.rand(y_n, x_n) - 0.5
        self.b = np.random.rand(y_n, 1) - 0.5

    def fforward(self, x):
        self.input = x
        return self.w.dot(x) + self.b

    def bprop(self, gradient, lr, batch_size):
        self.b = self.b - (1/batch_size * gradient * lr)
        self.w = self.w - (1/batch_size * gradient.dot(self.input.T) * lr)
        return self.w.T.dot(gradient)
