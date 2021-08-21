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

        self.activation = False

    """ OVERFLOW ERROR """
    def ff(self, data, activation=True):
        """
        Calculates the output of a single Layer with the according weights, biases and provided data
        :param data: The Contents/ Output of the previous Layer
        :param activation: Whether the Output should be transformed via an Activation-Function
        :return: Output of the Layer in numpy.arr-format
        """
        self.activation = activation
        a = np.matmul(data, self.w) + self.b
        self.z = a
        # overflow in power
        if activation:
            self.output = np.log(1 + np.power(math.e, a))
            return self.output
        self.output = a
        return self.output
