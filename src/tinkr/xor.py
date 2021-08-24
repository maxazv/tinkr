from dense import Dense
from activations import Tanh
from loss import mse, mse_prime

import numpy as np

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

epochs = 10000
lr = 0.1

tinkr = [
    Dense(2, 3),
    Tanh(),
    Dense(3, 1),
    Tanh()
]
# train
for e in range(epochs):
    error = 0
    for x, y in zip(X, Y):
        # feedforward
        output = x
        for layer in tinkr:
            output = layer.fforward(output)

        error += mse(y, output)

        # backpropagation
        local_gradient = mse_prime(y, output)
        for layer in reversed(tinkr):
            local_gradient = layer.bprop(local_gradient, lr)

    error /= len(X)
    print(f"{e + 1}/{epochs}, error={error}")

# prediction
print()
for x in X:
    z = x
    for layer in tinkr:
        z = layer.fforward(z)
    print(x.tolist(), "->", z)
