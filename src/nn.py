import numpy as np
import _ctypes
import math

class NeuralNet:
    def __init__(self, layer_shapes, activation='sigmoid'):
        """
        :param layer_shapes: layer_shapes[0] represents the input-layer,
        layer_shapes[1: len(layer_shapes)-2] represents the hidden-layers,
        layer_shapes[len(layer_shapes)-1] represents the output-layer
        """
        self.size = len(layer_shapes)

        self.layers = [np.zeros((layer_shapes[i], 1)) for i in range(self.size)]
        self.input = _ctypes.PyObj_FromPtr(id(self.layers[0]))
        self.output = _ctypes.PyObj_FromPtr(id(self.layers[self.size-1]))

        self.activations = [np.zeros((layer_shapes[i+1], 1)) for i in range(self.size-1)]

        self.weights = [np.random.normal(0, 1, size=(m.size, n.size)) for m, n in zip(self.layers[1:], self.layers[:-1])]
        self.biases = [np.zeros((layer_shapes[i+1], 1)) for i in range(self.size-1)]

        self.learning_rate = 0.5

        self.d_activations = []
        self.d_weights = []
        self.d_biases = []

    @staticmethod
    def sigmoid(x, prime=False):
        if not prime:
            return 1 / (1 + pow(math.e, -x))
        return x * (1 - x)

    def loss_func(self, truth, prime=False):
        if not prime:
            return pow((truth - self.layers[self.size-1]), 2)
        return -2 * (truth - self.layers[self.size-1])

    def guess(self, input_data):
        """
        Calculates output based on weights and biases and stores result in self.layers (feedforward)
        :param input_data: Data to be processed by the Neural Network
        """
        input_data = np.array([input_data])
        self.layers[0] = input_data.T

        for i in range(self.size-1):
            self.layers[i+1] = np.matmul(self.weights[i], self.layers[i])
            self.layers[i+1] = np.add(self.biases[i], self.layers[i+1])
            self.activations[i] = self.layers[i+1]      # save for gradient
            self.layers[i+1] = self.sigmoid(self.layers[i+1])

        return self.layers[self.size-1]

    def backprop(self, truth):
        """
        Calculates Gradients using the Chain-Rule
        :param truth: expected output of Neural Network in List-format [1, 0, ...]
        """
        # conversion from list to (n, 1) numpy-array
        self.d_weights = [np.zeros((m.size, n.size)) for m, n in zip(self.layers[1:], self.layers[:-1])]
        self.d_biases = [np.zeros((self.layers[i+1].shape[0], 1)) for i in range(self.size-1)]

        truth = np.array([truth]).T
        delta = self.loss_func(truth, True)

        for i in range(self.size, 2, -1):
            delta *= self.sigmoid(self.activations[i-2], True)
            self.d_biases[i-2] = delta
            self.d_weights[i-2] = np.matmul(delta, self.layers[i-2].T)
            delta = np.matmul(self.weights[i-2].T, delta)

        # add first layer d_weights
        delta *= self.sigmoid(self.activations[0], True)
        self.d_biases[0] = delta
        self.d_weights[0] = np.matmul(delta, self.layers[0].T)

    def calc_stepsize(self):
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        for i in range(len(self.weights)-1):
            self.weights[i] -= self.d_weights[i] * self.learning_rate
            self.biases[i] -= self.d_biases[i] * self.learning_rate

    def train(self):
        #lowest_val = 0.001
        #max_epochs = 1000
        print(self.layers)
        pass
