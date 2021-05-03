import os
from pathlib import Path
import numpy as np
from random import shuffle

class NeuralNetwork:
    """Create an instance of the neural network"""
    def __init__(self, step=0.01):

        self.parameters = NeuralNetwork.ReadNeuralNetwork() + NeuralNetwork.GenerateRandomNetwork() * step

        # self.parameters = NeuralNetwork.GenerateRandomNetwork()
        # NeuralNetwork.SaveNeuralNetwork(self.parameters)

    def Predict(self, a0):
        """Predict the result of a single sample and set the layer values"""
        z1 = np.matmul(self.parameters[0], a0) + self.parameters[3]
        a1 = self.Activation(z1)    # First hidden layer

        z2 = np.matmul(self.parameters[1], a1) + self.parameters[4]
        a2 = self.Activation(z2)  # Second hidden layer

        z3 = np.matmul(self.parameters[2], a2) + self.parameters[5]
        a3 = self.Activation(z3)  # Second hidden layer
        
        return np.around(a3)    

    @staticmethod
    def SaveNeuralNetwork(parameters):
        """Save the neural network in a file"""
        np.savez("neural-network-parameters.npz", parameters)

    @staticmethod
    def ReadNeuralNetwork():
        """Read the neural network from a file"""
        data = np.load("neural-network-parameters.npz", allow_pickle=True)

        parameters = np.array([data['arr_0'][0], data['arr_0'][1], data['arr_0'][2], \
                               data['arr_0'][3], data['arr_0'][4], data['arr_0'][5]], dtype=object)

        return parameters

    @staticmethod
    def GenerateRandomNetwork():
        layer_sizes        = [8, 200, 100, 4]
        weight_shapes      = [(a, b) for a, b in zip(layer_sizes[1:], layer_sizes[:-1])]

        weights            = np.array([np.random.standard_normal(weight_shape)/(weight_shape[1]**.5) for weight_shape in weight_shapes], dtype=object)
        biases             = np.array([np.random.standard_normal((layer_size, 1)) for layer_size in layer_sizes[1:]], dtype=object)
        parameters         = np.array([weights[0], weights[1], weights[2], biases[0], biases[1], biases[2]], dtype=object)

        return parameters

    @staticmethod
    def Activation(x):
        """Sigmoid activation function"""
        return 1/(1 + np.exp(-x))

# network = NeuralNetwork(1)
# print(network.Predict([[5], [5], [5], [5], [5], [5], [5], [5]]))
