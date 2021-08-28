""" Thanks https://github.com/TheIndependentCode/Neural-Network/tree/b97496c9a776cfd5e8d91cdef64e562766716342
    Had enough fun with understanding/ implementing MLPs completely on my own (First Branches represent my own designs/
    suffering) and decided to watch his video:
    https://www.youtube.com/watch?v=Lakz2MoHy6o&list=PL_hc17JNWhdAxb1meCSmBvhF4u8RkHYLy&index=12&t=1s
    to understand the next most fundamental Machine Learning algorithm without going through all the hustle again
    (To simply gain the knowledge and maybe implement my own optimisations/ ideas)."""
import numpy as np
from scipy import signal
from layer import Layer

class Conv(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)

        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def fforward(self, x):
        self.input = x
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output

    def bprop(self, gradient, lr, batch_size):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(gradient[i], self.kernels[i, j], "full")

        self.kernels -= lr * kernels_gradient
        self.biases -= lr * gradient
        return input_gradient
