import numpy as np
from activation import Activation

class ReLU(Activation):
    def __init__(self):
        def relu(x):
            return np.maximum(0, x)

        def relu_prime(x):
            return x > 0

        super().__init__(relu, relu_prime)

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)

class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)

class Softmax(Activation):
    def __init__(self):
        def softmax(x):
            exps = np.exp(x - x.max())
            return exps / sum(np.exp(x))

        def softmax_prime(x):
            sm = softmax(x)
            return sm * (1 - sm)

        super().__init__(softmax, softmax_prime)
