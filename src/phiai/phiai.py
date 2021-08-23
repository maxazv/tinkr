import numpy as np
from src.phiai.layer import Layer
from src.data_config import DataConfig

class PhiAI:
    def __init__(self, layer_shapes, last_activation=False, batch_size=1, lr=0.15):
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
            #dW = 1 / Y.size * np.matmul(local_gradient, self.layers[i-1].output.T)
            self.layers[i].b = self.layers[i].b - dB * self.lr
            self.layers[i].w = self.layers[i].w - dW * self.lr

            local_gradient = self.layers[i].w.T.dot(local_gradient)  # * Layer.relu(self.layers[i].z, True)

        dB = 1 / Y.size * np.sum(local_gradient)
        dW = 1 / Y.size * local_gradient.dot(X.T)
        self.layers[0].b = self.layers[0].b - dB * self.lr
        self.layers[0].w = self.layers[0].w - dW * self.lr

    def train(self, X, Y, epochs=1000, verbose=10):
        acc = 0
        for i in range(epochs):
            self.predict(X)
            self.backprop(X, Y)
            if i % verbose == 0:
                print('\nEpoch: ', i)
                acc = self.accuracy(self.argmax(self.layers[self.size-1].output), Y)
                print('Accuracy: ', acc)
        return self.model_to_list(), acc

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

    def model_to_list(self):
        model = []
        for i in range(self.size):
            model.append(self.layers[i].w)
            model.append(self.layers[i].b)
        return model

    def save_model(self, path='res/models.npz'):
        model = self.model_to_list()
        np.savez(path, *model)

    def load_model(self, path='res/models.npz'):
        model = np.load(path)
        key = 'arr_'
        for i in range(self.size):
            self.layers[i].w = model[key + str(i*2)]
            self.layers[i].b = model[key + str((i*2)+1)]
