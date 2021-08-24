import numpy as np
from layer import Layer

class Activation(Layer):
    def __init__(self, activate, a_prime):
        self.activate = activate
        self.a_prime = a_prime

    def fforward(self, x):
        self.input = x
        return self.activate(x)

    def bprop(self, gradient, lr):
        return np.multiply(gradient, self.a_prime(self.input))
