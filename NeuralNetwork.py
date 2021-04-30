import os
from pathlib import Path
import numpy as np
from random import shuffle

class NeuralNetwork:
    """Create an instance of the neural network"""
    def __init__(self):

        # self.parameters = np.array([self.weights[0], self.weights[1], self.biases[0], self.biases[1]], dtype=object)
        # self.GenerateRandomNetwork()

        # self.parameters += self.ReadNeuralNetwork()

        self.parameters = NeuralNetwork.ReadNeuralNetwork() + NeuralNetwork.GenerateRandomNetwork() / 10
        # self.parameters = NeuralNetwork.GenerateRandomNetwork()
        # NeuralNetwork.SaveNeuralNetwork(self.parameters)

        # self.parameters += np.random.standard_normal(self.parameters.shape) / 10

    def Predict(self, a0):
        """Predict the result of a single sample and set the layer values"""
        z1 = np.matmul(self.parameters[0], a0) + self.parameters[2]
        a1 = self.Activation(z1)    # Hidden layer

        z2 = np.matmul(self.parameters[1], a1) + self.parameters[3]
        a2 = self.Activation(z2)  # Output layer
        
        return np.around(a2)    

    @staticmethod
    def SaveNeuralNetwork(parameters):
        """Save the neural network in a file"""
        np.savez("neural-network-parameters.npz", parameters)

    @staticmethod
    def ReadNeuralNetwork():
        """Read the neural network from a file"""
        data = np.load("neural-network-parameters.npz", allow_pickle=True)

        parameters = np.array([data['arr_0'][0], data['arr_0'][1], data['arr_0'][2], data['arr_0'][3]], dtype=object)

        return parameters

    @staticmethod
    def GenerateRandomNetwork():
        layer_sizes        = [7, 20, 4]
        weight_shapes      = [(a, b) for a, b in zip(layer_sizes[1:], layer_sizes[:-1])]

        weights            = np.array([np.random.standard_normal(weight_shapes[0])/(weight_shapes[0][1]**.5), \
                                  np.random.standard_normal(weight_shapes[1])/(weight_shapes[1][1]**.5)], dtype=object)
        biases             = np.array([np.random.standard_normal((layer_sizes[1], 1)), np.random.standard_normal((layer_sizes[2], 1))], dtype=object)
        parameters         = np.array([weights[0], weights[1], biases[0], biases[1]], dtype=object)

        return parameters

    @staticmethod
    def Activation(x):
        """Sigmoid activation function"""
        return 1/(1 + np.exp(-x))

# network = NeuralNetwork()
# print(network.Predict([[5], [5], [5], [5], [5], [5], [5]]))