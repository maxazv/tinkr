import numpy as np
import math


class Layer:
    def __init__(self, data_n, output_n):
        """
        Utilised by the PhiAI a single Layer which consists of Weights, Biases, "activated"-Value and an
        Output based on the previous named Parameters
        :param data_n: Amount of Neurons of (the previous) Layer
        :param output_n: Amount of Neurons of the next Layer
        """
        self.w = np.random.normal(0, 1, size=(data_n, output_n))
        self.b = np.zeros((1, output_n))
        self.z = 0
        self.output = 0
        self.form = (data_n, output_n)

    def ff(self, data, activation=True):
        """
        Calculates the output of a single Layer with the according weights, biases and provided data
        :param data: The Contents/ Output of the previous Layer
        :param activation: Whether the Output should be transformed via an Activation-Function
        :return: Output of the Layer in numpy.arr-format
        """
        a = np.matmul(data, self.w) + self.b
        self.z = a
        if activation:
            self.output = np.log(1 + math.e ** a)
            return self.output
        self.output = a
        return self.output


class PhiAI:
    def __init__(self, layer_shapes):
        """
        Neural Network Class - made by Max
        :param layer_shapes: Structure of the Neural Network in List-Format
        """
        self.size = len(layer_shapes) - 1
        self.last_input = -1
        self.layers = [Layer(layer_shapes[i], layer_shapes[i + 1]) for i in range(self.size)]
        self.output = self.layers[self.size-1].output

        self.lr = 0.05

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
        if data.shape[0] != self.layers[0].form[0]:
            print('[ERR]: Dimensions do not match')
            return -1

        self.last_input = data
        curr_output = data
        for i in range(len(self.layers)):
            if i == len(self.layers) - 1:
                return self.layers[i].ff(curr_output, False)

            curr_output = self.layers[i].ff(curr_output)
        self.output = self.layers[self.size - 1].output
        return curr_output

    """ TO BE OPTIMISED HEAVILY -> PROBLEM: Transpose/ Shapes of Matrices
     TEST WITH ORIGINAL PROVED METHOD 
     IMPLEMENT BATCH-GRADIENT-DESCENT """
    def adjust(self, target):
        """
        Adjusts Weights and Biases based on the calculated Gradients
        :param target: The expected Output for the last used Inputs
        """
        delta = -2 * (target - self.layers[self.size - 1].output)  # mult by d_activation if necessary
        for i in range(self.size - 1, 0, -1):
            if i == self.size - 1:
                self.layers[i].b -= delta * self.lr
            else:
                self.layers[i].b -= delta.T * self.lr

            self.layers[i].w -= delta * self.layers[i - 1].output.T * self.lr
            delta = np.matmul(self.layers[i].w, delta) * self.__activation(self.layers[i - 1].z.T, 'log', True)
            # original: delta = delta * self.layers[i].w * self.__activation(self.layers[i - 1].z.T, 'log', True)

        self.layers[0].b -= delta.T * self.lr
        self.layers[0].w -= (delta * self.last_input).T * self.lr

        return True

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
                    print('REACHED')
                    return
                np.random.shuffle(training)
            c += 1

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
        for i in range(self.size-1):
            model.append((self.layers[i].w, self.layers[i].b))

        return model

    def train_digit(self, train_x, train_y, max_epochs=500, lowest_err=0.001):
        err, c = 0, 0
        for i in range(max_epochs):
            self.predict(train_x[i])
            self.adjust(train_y[i])
            err += (train_y[i] - self.layers[self.size - 1].output) ** 2
            if c % 5 == 0:
                print('Iteration: ', c)
                if err < 0.1:
                    return True
                err = 0
