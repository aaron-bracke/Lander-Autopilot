import os
import copy
from pathlib import Path
import numpy as np
from random import shuffle, getrandbits, randrange

class NeuralNetwork:
    """Create an instance of the neural network"""
    def __init__(self):
        self.parameters = NeuralNetwork.GenerateRandomNetwork()

    def Predict(self, a0):
        """Predict the result of a single sample and set the layer values"""
        z1 = np.matmul(self.parameters[0], a0) + self.parameters[3]
        a1 = self.Sigmoid(z1)    # First hidden layer

        z2 = np.matmul(self.parameters[1], a1) + self.parameters[4]
        a2 = self.Sigmoid(z2)  # Second hidden layer

        z3 = np.matmul(self.parameters[2], a2) + self.parameters[5]
        a3 = self.Sigmoid(z3)  # Output layer

        return np.around(a3.flatten())

    @staticmethod
    def SaveNeuralNetwork(parameters):
        """Save the neural network in a file"""
        np.savez("neural-network-parameters.npz", parameters)

    @staticmethod
    def ReadNeuralNetwork():
        """Read the neural network from a file"""
        parameters = np.load("neural-network-parameters.npz", allow_pickle=True)['arr_0']
        return parameters

    @staticmethod
    def GenerateRandomNetwork():
        layer_sizes        = [9, 16, 16, 4]
        weight_shapes      = [(a, b) for a, b in zip(layer_sizes[1:], layer_sizes[:-1])]

        weights            = np.array([np.random.standard_normal(weight_shape)/(weight_shape[1]**.5) for weight_shape in weight_shapes], dtype=object)
        biases             = np.array([np.random.standard_normal((layer_size, 1)) for layer_size in layer_sizes[1:]], dtype=object)
        parameters         = np.array([weights[0], weights[1], weights[2], biases[0], biases[1], biases[2]], dtype=object)

        return parameters

    @staticmethod
    def Sigmoid(x):
        """Sigmoid activation function"""
        return 1/(1 + np.exp(-x))

    @staticmethod
    def HypTan(x):
        """Hyperbolic tangent activation function"""
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

    @staticmethod
    def Gaussian(x):
        """Gaussian activation function"""
        return np.exp(-(x**2)/2.0)

    @staticmethod
    def CombineGenes(network1, network2):
        parameters1 = copy.deepcopy(network1.parameters)
        parameters2 = copy.deepcopy(network2.parameters)

        parameters = copy.deepcopy(parameters1)

        for i in range(parameters1.size):
            for j in range(parameters1[i].shape[0]):
                for k in range(parameters1[i].shape[1]):
                    if bool(getrandbits(1)):
                        parameters[i][j][k] = parameters1[i][j][k]
                    else:
                        parameters[i][j][k] = parameters2[i][j][k]
                    if randrange(100) < 4:
                        parameters[i][j][k] += np.random.standard_normal()/2.0

                    # if parameters[i][j][k] != parameters1[i][j][k] and parameters[i][j][k] != parameters2[i][j][k]:
                    #     print("MUTATION")
                    # elif parameters[i][j][k] == parameters1[i][j][k] and parameters[i][j][k] != parameters2[i][j][k]:
                    #     print("PARENT 1")
                    # elif parameters[i][j][k] == parameters2[i][j][k] and parameters[i][j][k] != parameters1[i][j][k]:
                    #     print("PARENT 2")
                    # else:
                    #     print("INVALID")

        return parameters

    def ResetNetwork(self):
        self.parameters = NeuralNetwork.GenerateRandomNetwork()
        NeuralNetwork.SaveNeuralNetwork(self.parameters)

# network = NeuralNetwork(True)
# network.ResetNetwork()
# print(network.Predict([[5], [5], [5], [5], [5], [5], [5], [5], [5]]))
