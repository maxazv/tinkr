import math
import numpy as np
from src.phiai.phiai import PhiAI
from src.data_config import DataConfig

def delta_log(x):
    return math.e ** x / (1 + math.e ** x)


'''
def test_ai(ptest, verbose=0, testcase=0):
    cl_test = ptest

    # phiAI Class
    t_data = [[0, 0],
              [0.5, 1],
              [1, 0]]
    test_i = testcase

    if cl_test == 0:
        # --[ Feedforward Algorithm ]--
        X = [0, 0.5, 1]
        Y_H = [0, 1, 0]
        k = testcase

        h = Layer(1, 2)
        y = Layer(2, 1)
        lr = 0.075

        h.ff(np.array([X[k]]))
        y.ff(h.output, False)
        l_v1 = y.output

        # --[ Loss Func ]--

        sr = (Y_H[k] - y.output) ** 2
        l_v2 = sr

        # --[ Update Params via Gradients ]--

        # --gradient output bias
        delta_sr = -2 * (Y_H[k] - y.output)
        delta_y_b = delta_sr
        # --update params
        y.b -= delta_y_b * lr
        # --more gradients and updates for y
        delta_y_w = delta_sr * h.output.T
        y.w -= delta_y_w * lr

        # ++gradients and updates for h
        delta_h_b = delta_sr * y.w * delta_log(h.z.T)
        delta_h_w = delta_sr * y.w * delta_log(h.z.T) * np.array([X[k]])
        h.b -= delta_h_b.T * lr
        h.w -= delta_h_w.T * lr
        h.ff(np.array([X[k]]))
        y.ff(h.output, False)
        l_v3 = y.output
        l_v4 = (Y_H[k] - y.output) ** 2

        if verbose > 0:
            print(f'First Guess: {l_v1}')
            print(f'Expected Output: [{Y_H[k]}]')
            print(f'     -> Error: {l_v2}')
            print('--------------------------------')
            print(f'Adjusted Guess: {l_v3}')
            print(f'     -> Error: {l_v4}\n')

    elif cl_test == 1:
        """ -***-[ Neural Network Class ]-***- """
        nn = PhiAI([1, 2, 1])

        # Test phiai Class
        # + first prediction
        nn.predict(np.array([t_data[test_i][0]]))
        n_v1 = nn.layers[nn.size - 1].output
        n_v2 = nn.loss(t_data[test_i][1])
        # + adjust params once
        nn.adjust(t_data[test_i][1])
        nn.predict(np.array([t_data[test_i][0]]))
        n_v3 = nn.layers[nn.size - 1].output
        n_v4 = nn.loss(t_data[test_i][1])

        if verbose > 0:
            print(f'First Guess: {n_v1}')
            print(f'Expected Output: {t_data[test_i][1]}')
            print(f'     -> Error: {n_v2}')
            print('--------------------------------')
            print(f'Adjusted Guess: {n_v3}')
            print(f'     -> Error: {n_v4}')

        if verbose > 1:
            nn.train(t_data)
            nn.predict(np.array([t_data[0][0]]))
            print('--------------------------------')
            print("\nExpected:", t_data[0][1])
            print("Predicted:", nn.loss(t_data[0][1]))

            nn.predict(np.array([t_data[1][0]]))
            print("\nExpected:", t_data[1][1])
            print("Predicted:", nn.loss(t_data[1][1]))

            nn.predict(np.array([t_data[2][0]]))
            print("\nExpected:", t_data[2][1])
            print("Predicted:", nn.loss(t_data[2][1]))

    elif cl_test == 2:
        nnv2 = PhiAI([1, 2, 2, 1])

        nnv2.predict(np.array([t_data[test_i][0]]))
        print(f'First Guess: {nnv2.layers[nnv2.size-1].output}')
        print(f'Expected: {t_data[test_i][1]}')
        print(f'First Guess Error: {nnv2.loss(t_data[test_i][1])}')
        nnv2.adjust(t_data[test_i][1])
        nnv2.predict(np.array([t_data[test_i][0]]))
        print('--------------------------------')
        print(f'Adjusted Guess: {nnv2.layers[nnv2.size - 1].output}')
        print(f'Expected: {t_data[test_i][1]}')
        print(f'Adjusted Guess Error: {nnv2.loss(t_data[test_i][1])}')

        if verbose > 1:
            nnv2.train(t_data)
            nnv2.predict(np.array([t_data[0][0]]))
            print('--------------------------------')
            print("\nExpected:", t_data[0][1])
            print("Predicted:", nnv2.loss(t_data[0][1]))

            nnv2.predict(np.array([t_data[1][0]]))
            print("\nExpected:", t_data[1][1])
            print("Predicted:", nnv2.loss(t_data[1][1]))

            nnv2.predict(np.array([t_data[2][0]]))
            print("\nExpected:", t_data[2][1])
            print("Predicted:", nnv2.loss(t_data[2][1]))
'''
dc = DataConfig()
dc.load()
X_train, Y_train = dc.setup_data()
X_train = dc.normalize(X_train, 255.)


def test_digit_ai():
    phiai = PhiAI([784, 10, 10], lr=0.15)
    phiai.train(X_train, Y_train, epochs=750)
    phiai.save_model()


def test_trained_model(path, idx=0):
    trained = PhiAI([784, 10, 10])
    trained.load_model(path)

    img = X_train[:, idx, None]
    dc.plot_data(img)
    print("Expected: ", Y_train[idx])
    trained.predict(img)
    predict = trained.argmax(trained.layers[trained.size-1].output)
    print("Predicted: ", predict)


if __name__ == '__main__':
    #test_digit_ai()
    test_trained_model('C:/Users/Asus/Desktop/models/86.npz')
    '''
    digit = 0
    if digit == 0:
        test_digit_ai()
    else:
        test = 2
        verbose_level = 1
        test_case = 0
        test_ai(test, verbose_level, test_case)
    '''
