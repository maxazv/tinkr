import numpy as np

class Activation:
    def __init__(self, a, a_prime):
        self.a = a
        self.a_prime = a_prime

    def activate(self, x):
        self.input = x
        return self.activate(x)

    def backprop(self, x):
        return np.multiply()
