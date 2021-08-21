import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class DigitData:
    def __init__(self):
        self.data, self.form = 0, (0, 0)
        self.data_dev, self.Y_dev, self.X_dev = (0, )*3

        self.path = ''

    @staticmethod
    def one_hot(Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        return one_hot_Y

    @staticmethod
    def plot_data(d):
        plt.figure(figsize=(7, 7))

        grid_data = d.reshape(28, 28)
        plt.imshow(grid_data, interpolation='none', cmap='gray')
        plt.show()

    @staticmethod
    def normalize(dt, val):
        dt = dt/val
        return dt

    def load(self, path='res/train.csv'):
        self.path = path
        self.data = pd.read_csv(self.path)
        self.data = np.array(self.data)
        self.form = self.data.shape
        np.random.shuffle(self.data)

    def setup_data(self, j=1000, k=21000):
        self.data_dev = self.data[0:j].T
        self.Y_dev = self.data_dev[0]
        self.X_dev = self.data_dev[1:self.form[1]]

        data_train = self.data[j:k].T
        Y_train = data_train[0]
        X_train = data_train[1:self.form[1]]

        return Y_train, X_train.T
