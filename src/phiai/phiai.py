import numpy as np
from src.phiai.layer import Layer
from src.data_config import DataConfig

class PhiAI:
    def __init__(self, layer_shapes, last_activation=False, batch_size=1, lr=0.1):
        """Neural Network Class with layer-format described by `layer_shapes` - made by maxazv"""
        self.size = len(layer_shapes) - 1
        self.last_input = None
        self.layers = [Layer(layer_shapes[i], layer_shapes[i+1]) for i in range(self.size)]

        self.output = self.layers[self.size-1].output

        self.lr = lr
        self.last_activation = last_activation
        self.batch_size = batch_size

    def loss(self, Y):
        return self.layers[self.size-1].output - Y

    def predict(self, data):
        """Calculates output based on weights and biases and stores result in self.layers (feedforward)"""
        self.last_input = data
        curr_output = data
        for i in range(len(self.layers)):
            if i == len(self.layers)-1:
                return self.layers[i].ff(curr_output, 'softmax')
            curr_output = self.layers[i].ff(curr_output, 'relu')
        return curr_output

    def calc_stepsize(self, idx, dB, dW):
        self.layers[idx].b = self.layers[idx].b - dB * self.lr
        self.layers[idx].w = self.layers[idx].w - dW * self.lr

    def backprop(self, X, Y):
        local_gradient = self.loss(Y)
        for i in range(self.size-1, 0, -1):
            dB = 1 / Y.size * np.sum(local_gradient, 1)
            dW = 1 / Y.size * local_gradient.dot(self.layers[i-1].output.T)
            self.calc_stepsize(i, np.array([dB]).T, dW)

            local_gradient = self.layers[i].w.T.dot(local_gradient) * Layer.relu(self.layers[i].z, True)

        dB = 1 / Y.size * np.sum(local_gradient, 1)
        dW = 1 / Y.size * local_gradient.dot(X.T)
        self.calc_stepsize(0, np.array([dB]).T, dW)

    def train(self, X, Y, epochs=500):
        for i in range(epochs):
            self.predict(X)
            self.backprop(X, Y)
            if i % 15 == 0:
                print('\nEpoch: ', i)
                print('Accuracy: ', self.accuracy(self.argmax(self.layers[self.size-1].output), Y))
        return self.pull_model()

    @staticmethod
    def argmax(A):
        return np.argmax(A, 0)

    @staticmethod
    def accuracy(predicted, Y):
        print(predicted, "\n", Y)
        return np.sum(predicted == Y) / Y.size

    def minibatch_gd(self, y_train, x_train, max_epochs=1000):
        minibatches = DataConfig().create_batches(y_train, x_train, self.batch_size)
        mse = 100
        for i in range(len(minibatches)):
            if i > max_epochs:
                return
            if mse < 0.01:
                print('REACHED')
                return
            self.backprop(np.array(minibatches[i][1]), np.array(minibatches[i][0]))
            mse = np.sum(1/self.batch_size*self.loss(np.array([minibatches[i][0]])))

    # load current nn model
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

    # extract current nn model
    def pull_model(self):
        model = []
        for i in range(self.size):
            model.append((self.layers[i].w, self.layers[i].b))
        return model
