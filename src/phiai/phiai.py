import numpy as np
from src.phiai.layer import Layer
from src.data_config import DigitData

class PhiAI:
    def __init__(self, layer_shapes, last_activation=False, batch_size=1, lr=0.05, af='log'):
        """Neural Network Class with layer-format described by `layer_shapes` - made by maxazv"""
        self.size = len(layer_shapes) - 1
        self.last_input = None
        self.layers = [Layer(layer_shapes[i], layer_shapes[i+1]) for i in range(self.size)]

        self.output = self.layers[self.size-1].output

        self.lr = lr
        self.last_activation = last_activation
        self.af = af
        self.batch_size = batch_size

        self.digit_data = DigitData()

    @staticmethod
    def __activation(x, method='log', prime=True):
        if prime:
            if method == 'log':
                return np.exp(x) / (1 + np.exp(x))
            elif method == 'sigmoid':
                return x * (1 - x)
        return 0

    @staticmethod
    def create_batches(y_t, x_t, batch_size):
        minibatches = []
        y_batch, x_batch = None, None
        for i in range(y_t.shape[0] // batch_size):
            y_batch = y_t[i * batch_size:(i + 1) * batch_size]
            x_batch = x_t[i * batch_size:(i + 1) * batch_size]
            minibatches.append([y_batch, x_batch])

        if y_t.size % batch_size != 0:
            minibatches.append([y_batch, x_batch])
        return minibatches

    def loss(self, target):
        return (target - self.layers[self.size - 1].output) ** 2

    def predict(self, data):
        """Calculates output based on weights and biases and stores result in self.layers (feedforward)"""
        self.last_input = data
        curr_output = data
        for i in range(len(self.layers)):
            if i == len(self.layers)-1:
                return self.layers[i].ff(curr_output, self.last_activation, self.af)

            curr_output = self.layers[i].ff(curr_output, af=self.af)
        self.output = self.layers[self.size - 1].output
        return curr_output

    # train and default train model
    def adjust(self, y_train, x_train=None, method='stochastic'):
        """Adjusts Weights and Biases based on the calculated Gradients with `method`"""
        if method == 'stochastic':
            self.backprop(y_train)
        elif method == 'minibatch':
            self.minibatch_gd(y_train, x_train, self.batch_size)

    def backprop(self, target, stochastic=True):
        delta = -2 * (target - self.layers[self.size - 1].output)
        if self.layers[self.size - 1].activation:
            delta *= self.__activation(self.layers[self.size - 1].z.T, self.af, True).T

        for i in range(self.size - 1, 0, -1):
            if stochastic:
                self.layers[i].b -= delta * self.lr
                self.layers[i].w -= np.matmul(delta.T, self.layers[i - 1].output).T * self.lr
            else:
                delta_b = delta * self.lr
                self.layers[i].b -= np.array([(1/delta.shape[0]) * np.sum(delta_b, axis=0)])
                delta_w = np.zeros(self.layers[i].w.shape)
                for j in range(delta.shape[0]):
                    tnsp_delta = np.array([delta[j]]).T
                    tnsp_output = np.array([self.layers[i-1].output[j]])
                    delta_w += np.matmul(tnsp_delta, tnsp_output).T * self.lr
                self.layers[i].w -= (1/delta.shape[0]) * delta_w

            delta = np.matmul(self.layers[i].w, delta.T).T * self.__activation(self.layers[i - 1].z.T, self.af, True).T

        if stochastic:
            self.layers[0].b -= delta * self.lr
            self.layers[0].w -= np.matmul(delta.T, self.last_input).T * self.lr
        else:
            delta_b = delta * self.lr
            self.layers[0].b -= np.array([(1/delta.shape[0]) * np.sum(delta_b, axis=0)])
            delta_w = np.zeros(self.layers[0].w.shape)
            for j in range(delta.shape[0]):
                tnsp_delta = np.array([delta[j]]).T
                tnsp_output = np.array([self.last_input[j]])
                delta_w += np.matmul(tnsp_delta, tnsp_output).T * self.lr
            self.layers[0].w -= (1/delta.shape[0]) * delta_w
        return True

    def minibatch_gd(self, y_train, x_train, max_epochs=80000):
        minibatches = self.create_batches(y_train, x_train, self.batch_size)
        mse = 100
        print("Epochs: ", len(minibatches))
        for i in range(len(minibatches)):
            if i > max_epochs:
                return
            if mse < 0.01:
                print('REACHED')
                return
            self.predict(minibatches[i][1])
            self.backprop(np.array(minibatches[i][0]), False)
            mse = np.sum(1/self.batch_size*self.loss(np.array([minibatches[i][0]])))

    def train(self, training, max_epochs=250, lowest_err=0.01):
        """Utilises self.predict as well as self.adjust to optimise the Neural Network to the given data `training`"""
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
    def load_data(self, path='res/train.csv', j=1000, k=60000):
        self.digit_data.load(path)
        Y_t, X_t = self.digit_data.setup_data(j, k)
        train_data_y = DigitData.one_hot(Y_t)
        return train_data_y, X_t
