import os
import copy
from pathlib import Path
import numpy as np
from random import shuffle, getrandbits, randrange

class NeuralNetwork:
    """Create an instance of the neural network"""
    def __init__(self):
        self.parameters = NeuralNetwork.ReadNeuralNetwork()
        self.layers = np.array([np.zeros(layer_size) for layer_size in [9, 16, 16, 4]], dtype=object)
        self.zlayers = np.array([np.zeros(layer_size) for layer_size in [9, 16, 16, 4]], dtype=object)

    def Predict(self, a0):
        """Predict the result of a single sample and set the layer values"""
        z1 = np.matmul(self.parameters[0], a0) + self.parameters[3]
        a1 = self.Sigmoid(z1)    # First hidden layer

        z2 = np.matmul(self.parameters[1], a1) + self.parameters[4]
        a2 = self.Sigmoid(z2)  # Second hidden layer

        z3 = np.matmul(self.parameters[2], a2) + self.parameters[5]
        a3 = self.Sigmoid(z3)  # Output layer

        self.zlayers = np.array([a0, z1, z2, z3], dtype=object) # Save the results in arrays
        self.layers = np.array([a0, a1, a2, a3], dtype=object)

        return np.around(a3.flatten())
    
    def DetermineGradient(self, y):
        """Determine the gradient vector dC to the corresponding label y"""

        print(f"{np.linalg.norm((self.layers[3] - y)**2):0.01f}")

        common_factor3 = 2 * (self.layers[3] - y) * NeuralNetwork.dSigmoid(self.zlayers[3])
        common_factor2 = NeuralNetwork.dSigmoid(self.zlayers[2]) * np.dot(np.matmul(self.parameters[2], np.ones((len(self.layers[2]),1))).T, common_factor3)
        common_factor1 = NeuralNetwork.dSigmoid(self.zlayers[1]) * np.dot(np.matmul(self.parameters[1], np.ones((len(self.layers[1]),1))).T, common_factor2)

        
        dW1 = np.matmul(common_factor1, self.layers[0].T)
        dW2 = np.matmul(common_factor2, self.layers[1].T)
        dW3 = np.matmul(common_factor3, self.layers[2].T)

        db1 = common_factor1
        db2 = common_factor2
        db3 = common_factor3
        
        return copy.deepcopy(np.array([dW1, dW2, dW3, db1, db2, db3], dtype=object))

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
    def dSigmoid(x):
        """Derivative of Sigmoid activation function"""
        return NeuralNetwork.Sigmoid(x) * (1 - NeuralNetwork.Sigmoid(x))

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

# network = NeuralNetwork()
# print(network.Predict(np.array([[5], [5], [5], [5], [5], [5], [5], [5], [5]])))
# network.DetermineGradient(np.array([[1, 0, 0, 1]]).T)
