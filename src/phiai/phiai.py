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

    def predict(self, data):
        """Calculates output based on weights and biases and stores result in self.layers (feedforward)"""
        self.last_input = data
        curr_output = data
        for i in range(len(self.layers)):
            if i == len(self.layers)-1:
                return self.layers[i].ff(curr_output, 'softmax')
            curr_output = self.layers[i].ff(curr_output, 'relu')
        return curr_output

    def backprop(self, X, Y):
        """Adjusts Weights and Biases based on the Error/ Gradients"""
        onehot = DataConfig.one_hot(Y)
        local_gradient = self.layers[self.size-1].output - onehot
        for i in range(self.size-1, 0, -1):
            dB = 1 / Y.size * np.sum(local_gradient)
            dW = 1 / Y.size * local_gradient.dot(self.layers[i-1].output.T)
            self.layers[i].b = self.layers[i].b - dB * self.lr
            self.layers[i].w = self.layers[i].w - dW * self.lr

            local_gradient = self.layers[i].w.T.dot(local_gradient) * Layer.relu(self.layers[i].z, True)

        dB = 1 / Y.size * np.sum(local_gradient)
        dW = 1 / Y.size * local_gradient.dot(X.T)
        self.layers[0].b = self.layers[0].b - dB * self.lr
        self.layers[0].w = self.layers[0].w - dW * self.lr

    def train(self, X, Y, epochs=500):
        for i in range(epochs):
            self.predict(X)
            self.backprop(X, Y)
            if i % 10 == 0:
                print('\nEpoch: ', i)
                print('Accuracy: ', self.accuracy(self.argmax(self.layers[self.size-1].output), Y))
        return self.pull_model()

    @staticmethod
    def argmax(A):
        return np.argmax(A, 0)

    @staticmethod
    def accuracy(predicted, Y):
        print(predicted, Y)
        return np.sum(predicted == Y) / Y.size

    def minibatch_gd(self, y_train, x_train, max_epochs=1000):
        minibatches = DataConfig().create_batches(y_train, x_train, self.batch_size)
        for i in range(max_epochs):
            pass

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
