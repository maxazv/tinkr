import numpy as np

def mse(y_hat, y_pred):
    return np.mean(np.power(y_hat - y_pred, 2))

def mse_prime(y_hat, y_pred):
    return 2 * (y_pred - y_hat) / np.size(y_hat)

def binary_cross_entropy(y_hat, y_pred):
    return np.mean(-y_hat * np.log(y_pred) - (1 - y_hat) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_hat, y_pred):
    return ((1 - y_hat) / (1 - y_pred) - y_hat / y_pred) / np.size(y_hat)
