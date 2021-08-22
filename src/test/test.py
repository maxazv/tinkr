import numpy as np
import math
from src.phiai.phiai import PhiAI
from src.phiai.phiai import Layer

def delta_log(x):
    return math.e ** x / (1 + math.e ** x)


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

def test_digit_ai():
    digit_phi = PhiAI([784, 15, 15, 10], True, lr=0.00001, af='sigmoid')
    # configure target and corresponding data set
    Y, X = digit_phi.load_data()
    X = digit_phi.digit_data.normalize(X, 255)
    # plot data
    #digit_phi_batched.digit_data.plot_data(X[5])
    #print(Y[5])
    """ BATCHED """
    '''
    size = 32
    digit_phi_batched = PhiAI([784, 15, 15, 10], True, size, lr=0.00007)
    
    mb = PhiAI.create_batches(Y, X, size)
    digit_phi_batched.predict(mb[0][1])
    print(np.sum(1/size*digit_phi_batched.loss(mb[0][0])))
    # digit_phi_batched.backprop(np.array(mb[0][0]), False)
    #digit_phi_batched.predict(mb[0][1])
    #print(np.sum(1/10*digit_phi_batched.loss(mb[0][0])))
    digit_phi_batched.adjust(Y, X, 'minibatch')
    digit_phi_batched.predict(mb[0][1])
    print(np.sum(1/size*digit_phi_batched.loss(mb[0][0])))
    print('Random Predict: ', digit_phi_batched.predict(X[0]))
    print('Expect: ', Y[0])
    '''
    """ STOCHASTIC """
    #'''
    digit_phi.predict(np.array([X[0]]))
    print('Guessed: ', digit_phi.layers[digit_phi.size-1].output)
    print('Expected: ', Y[0])
    print('Error Sum: ', np.sum(digit_phi.loss(Y[0])))
    print('--------------------------------')
    digit_phi.adjust(Y[0])
    digit_phi.predict(np.array([X[0]]))
    print(f'Adjusted Guess: {digit_phi.layers[digit_phi.size-1].output}')
    print(f'Expected: {Y[0]}')
    print(f'Adjusted Guess Error: {np.sum(digit_phi.loss(Y[0]))}')
    #'''


if __name__ == '__main__':
    digit = 1
    if digit == 0:
        test_digit_ai()
    else:
        test = 2
        verbose_level = 1
        test_case = 0
        test_ai(test, verbose_level, test_case)
