import numpy as np
from layer import Layer

class Conv(Layer):
    def __init__(self, input_shape, filter_shape, depth):
        super(Conv, self).__init__()

    def fforward(self, x):
        return x

    def bprop(self, gradient, lr, batch_size):
        return gradient, lr, batch_size
