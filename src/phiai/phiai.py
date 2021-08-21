import numpy as np
import math
from src.phiai.layer import Layer
from src.data_config import DigitData

class PhiAI:
    def __init__(self, layer_shapes, last_activation=False):
        """
        Neural Network Class - made by Max
        :param layer_shapes: Structure of the Neural Network in List-Format
        """
        self.size = len(layer_shapes) - 1
        self.last_input = -1
        self.layers = [Layer(layer_shapes[i], layer_shapes[i + 1]) for i in range(self.size)]
        self.output = self.layers[self.size-1].output

        self.lr = 0.05
        self.last_activation = last_activation

        self.digit_data = DigitData()

    @staticmethod
    def __activation(x, method='log', prime=True):
        if prime:
            if method == 'log':
                return math.e ** x / (1 + math.e ** x)
            elif method == 'sigmoid':
                return x * (1 - x)
        return 0

    def loss(self, target):
        return (target - self.layers[self.size - 1].output) ** 2

    def predict(self, data):
        """
        Calculates output based on weights and biases and stores result in self.layers (feedforward)
        :param data: Data to be processed by Phi-Neural-Network
        :return: Returns the calculated output
        """
        self.last_input = data
        curr_output = data
        for i in range(len(self.layers)):
            if i == len(self.layers) - 1:
                return self.layers[i].ff(curr_output, self.last_activation)

            curr_output = self.layers[i].ff(curr_output)
        self.output = self.layers[self.size - 1].output
        return curr_output

    # train and default train model
    def adjust(self, y_train, x_train=None, method='stochastic', batch_size=10):
        """
        Adjusts Weights and Biases based on the calculated Gradients
        :param y_train: The expected Output for the last used Inputs
        :param x_train: Neural Network Input Data
        :param method: Gradient Descent Variant to use 'stochastic' or 'minibatch'
        :param batch_size: If minibatch Algorithm is used; batch size of minibatch
        """
        if method == 'stochastic':
            self.stochastic_gd(y_train)
        elif method == 'minibatch':
            self.minibatch_gd(y_train, x_train, batch_size)

    def stochastic_gd(self, target):
        delta = -2 * (target - self.layers[self.size - 1].output)
        if self.layers[self.size - 1].activation:
            delta *= self.__activation(self.layers[self.size - 1].z.T, 'log', True).T

        for i in range(self.size - 1, 0, -1):
            self.layers[i].b -= delta * self.lr
            self.layers[i].w -= np.matmul(delta.T, self.layers[i - 1].output).T * self.lr
            delta = np.matmul(self.layers[i].w, delta.T).T * self.__activation(self.layers[i - 1].z.T, 'log', True).T

        self.layers[0].b -= delta * self.lr
        self.layers[0].w -= np.matmul(delta.T, self.last_input).T * self.lr
        return True

    """ TO BE RESEARCHED/ IMPLEMENTED """
    def minibatch_gd(self, y_train, x_train, batch_size, max_epochs=1000):
        pass

    def train(self, training, max_epochs=250, lowest_err=0.01):
        """
        Utilises self.predict as well as self.adjust to optimise the Neural Network to the given data
        :param training: The training data in List-Format: [[Input, Expected], ...]
        :param max_epochs: The maximum training iterations before the Neural Network stops its training
        :param lowest_err: The lowest error value before the Neural Network stops its training
        """
        c, err, i = (0, ) * 3
        while c < max_epochs:
            self.adjust(training[i][1])
            self.predict(np.array([training[i][0]]))
            err += (training[i][0] - self.layers[self.size - 1].output) ** 2
            if (c % len(training)) == 0:
                i = 0
                err = 0
                if err < lowest_err:
                    return
                np.random.shuffle(training)
            c += 1

    # load and extract current nn model
    def load_model(self, layers):
        if len(self.layers) != len(layers):
            return False
        for i in range(len(layers)):
            shape = layers[i][0].shape
            if shape != self.layers[i].form:
                return False
            curr = Layer(shape[0], shape[1])
            curr.w = layers[i][0]
            curr.b = layers[i][1]
            self.layers[i] = curr
        return True

    def pull_model(self):
        model = []
        for i in range(self.size):
            model.append((self.layers[i].w, self.layers[i].b))
        return model

    # digit data configuration and training
    def load_data(self, path='res/train.csv'):
        self.digit_data.load(path)
        Y_t, X_t = self.digit_data.setup_data()
        train_data_y = DigitData.one_hot(Y_t)
        return train_data_y, X_t

    def train_digit(self, train_y, train_x, max_epochs=500, lowest_err=0.1):
        err, c = 0, 0
        for i in range(max_epochs):
            self.predict(train_x[i])
            self.adjust(train_y[i])
            err += (train_y[i] - self.layers[self.size - 1].output) ** 2
            if c % 5 == 0:
                print('Iteration: ', c)
                if err < lowest_err:
                    return -1
                err = 0
        return 1
