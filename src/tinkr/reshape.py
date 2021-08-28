import numpy as np
from layer import Layer

class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def fforward(self, x):
        return np.reshape(x, self.output_shape)

    def bprop(self, gradient, lr, batch_size):
        return np.reshape(gradient, self.input_shape)
