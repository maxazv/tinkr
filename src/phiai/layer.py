import numpy as np

class Layer:
    def __init__(self, data_n, output_n):
        """Single Neural Network Layer of shape (`data_n`, `output_n`)"""
        self.w = np.random.normal(0, 1, size=(data_n, output_n))
        self.b = np.zeros((1, output_n))
        self.z = 0
        self.output = 0

        self.form = (data_n, output_n)
        self.activation = False

    def ff(self, data, activation=True, af='log'):
        """Calculates the output of a single Layer with the according weights, biases and provided data"""
        self.activation = activation
        a = np.matmul(data, self.w) + self.b
        self.z = a
        if activation:
            if af == 'log':
                self.output = np.log(1 + np.exp(a))
                return self.output
            elif af == 'sigmoid':
                self.output = 1 / (1 + np.exp(-a))
                return self.output
        self.output = a

        return self.output
