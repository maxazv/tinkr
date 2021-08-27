import numpy as np
from layer import Layer

class Dropout(Layer):
    def __init__(self, training=False, rate=0.35):
        self.rate = rate
        self.training = training

    def fforward(self, x):
        out = x
        if self.training:
            mask = np.random.rand(x.shape[0], x.shape[1])
            d = 1 / (1 - self.rate)
            out = np.where(mask <= 0.2, 0, x * d)
            self.input = out / x
        return out

    def bprop(self, gradient, lr, batch_size=1):
        if self.training:
            return self.input * gradient
        return gradient
