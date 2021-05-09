import os
from pathlib import Path
import numpy as np
from random import shuffle, getrandbits, randrange

class NeuralNetwork:
    """Create an instance of the neural network"""
    def __init__(self, read_trained, parent1=None, parent2=None):
        # self.parameters = NeuralNetwork.GenerateRandomNetwork()
        # NeuralNetwork.SaveNeuralNetwork(self.parameters)

        if read_trained:
            self.parameters = NeuralNetwork.ReadNeuralNetwork()
        elif parent1 is not None and parent2 is not None:
            self.parameters = NeuralNetwork.CombineGenes(parent1, parent2)
        else:
            self.parameters = NeuralNetwork.GenerateRandomNetwork()

        # print(self.parameters[3], self.parameters.size)

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
        parameters = np.load("neural-network-parameters.npz", allow_pickle=True)['arr_0']
        return parameters

    @staticmethod
    def GenerateRandomNetwork():
        layer_sizes        = [9, 10, 10, 4]
        weight_shapes      = [(a, b) for a, b in zip(layer_sizes[1:], layer_sizes[:-1])]

        weights            = np.array([np.random.standard_normal(weight_shape)/(weight_shape[1]**.5) for weight_shape in weight_shapes], dtype=object)
        biases             = np.array([np.random.standard_normal((layer_size, 1)) for layer_size in layer_sizes[1:]], dtype=object)
        parameters         = np.array([weights[0], weights[1], weights[2], biases[0], biases[1], biases[2]], dtype=object)

        return parameters

    @staticmethod
    def Activation(x):
        """Sigmoid activation function"""
        return 1/(1 + np.exp(-x))

    @staticmethod
    def CombineGenes(network1, network2):
        parameters1 = network1.parameters
        parameters2 = network2.parameters

        parameters = np.copy(parameters1)
        for i in range(parameters.size):
            for j in range(parameters[i].shape[0]):
                for k in range(parameters[i].shape[1]):
                    if bool(getrandbits(1)):
                        parameters[i][j, k] = parameters2[i][j, k]
                    if randrange(100) < 5:
                        parameters[i][j, k] = np.random.standard_normal()

        return parameters


# network = NeuralNetwork(True)
# print(network.Predict([[5], [5], [5], [5], [5], [5], [5], [5], [5]]))
