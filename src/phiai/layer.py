import numpy as np

class Layer:
    def __init__(self, data_n, output_n):
        """Single Neural Network Layer of shape (`output_n`, `data_n`)"""
        self.w = np.random.rand(output_n, data_n) - 0.5
        self.b = np.random.rand(output_n, 1) - 0.5
        self.z = 0
        self.output = 0

        self.form = (output_n, data_n)
        self.activation = False

    @staticmethod
    def relu(Z, prime=False):
        if prime:
            return Z > 0
        return np.maximum(0, Z)

    @staticmethod
    def softmax(Z):
        return np.exp(Z) / sum(np.exp(Z))

    def ff(self, data, funct='relu', activation=True):
        """Calculates the output of a single Layer with the according weights, biases and provided data"""
        self.activation = activation

        self.z = self.w.dot(data) + self.b
        if activation:
            if funct == 'relu':
                self.output = self.relu(self.z)
            if funct == 'softmax':
                self.output = self.softmax(self.z)
        return self.output
