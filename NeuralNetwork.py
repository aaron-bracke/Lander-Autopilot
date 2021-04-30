import os
from pathlib import Path
import numpy as np
from random import shuffle

class NeuralNetwork:
    """Create an instance of the neural network"""
    def __init__(self):
        np.random.seed(7)
        self.layer_sizes        = [7, 20, 4]
        self.layers             = np.array([np.zeros(layer_size) for layer_size in self.layer_sizes], dtype=object)
        self.zlayers            = np.array([np.zeros(layer_size) for layer_size in self.layer_sizes], dtype=object)
        self.weight_shapes      = [(a, b) for a, b in zip(self.layer_sizes[1:], self.layer_sizes[:-1])]

        self.weights            = np.array([np.random.standard_normal(self.weight_shapes[0])/(self.weight_shapes[0][1]**.5), \
                                  np.random.standard_normal(self.weight_shapes[1])/(self.weight_shapes[1][1]**.5)], dtype=object)
        self.biases             = np.array([np.zeros((self.layer_sizes[1], 1)), np.zeros((self.layer_sizes[2], 1))], dtype=object)

        self.parameters         = np.array([self.weights[0], self.weights[1], self.biases[0], self.biases[1]], dtype=object)

    def Predict(self, a0):
        """Predict the result of a single sample and set the layer values"""
        try: 
            a0 = np.squeeze(a0, axis=2).T
        except Exception:
            pass
        z1 = np.matmul(self.weights[0], a0) + self.biases[0]
        a1 = self.Activation(z1)    # Hidden layer

        z2 = np.matmul(self.weights[1], a1) + self.biases[1]
        a2 = self.Activation(z2)  # Output layer
        
        return a2    

    @staticmethod
    def Activation(x):
        """Sigmoid activation function"""
        return 1/(1 + np.exp(-x))

    def Identify(self, image_name, mode):
        """Identify a given image"""
        if mode == "fleet":
            input_data = self.RetrieveData(False, self.image_resolution, is_training=False, image_name=image_name)
            parameters = np.load("fleetparameters.npz", allow_pickle=True)

            np.put(self.weights1, 0, parameters['arr_0'][0]), np.put(self.weights1, 1, parameters['arr_0'][1])
            np.put(self.weights1, 2, parameters['arr_0'][2]), np.put(self.biases, 1, parameters['arr_0'][3])
            np.put(self.weights0, 0, parameters['arr_0'][4]), np.put(self.weights0, 1, parameters['arr_0'][5])
            np.put(self.biases, 0, parameters['arr_0'][6])

            print("\n Prediction:", fleet_list[np.argmax(self.Predict(input_data))])
        elif mode == "type":
            input_data = self.RetrieveData(True, self.image_resolution, is_training=False, image_name=image_name)
            parameters = np.load("typeparameters.npz", allow_pickle=True)

            np.put(self.weights1, 0, parameters['arr_0'][0]), np.put(self.weights1, 1, parameters['arr_0'][1])
            np.put(self.weights1, 2, parameters['arr_0'][2]), np.put(self.biases, 1, parameters['arr_0'][3])
            np.put(self.weights0, 0, parameters['arr_0'][4]), np.put(self.weights0, 1, parameters['arr_0'][5])
            np.put(self.biases, 0, parameters['arr_0'][6])

            print(type_list[np.argmax(self.Predict(input_data))])

type_list = ["Boeing 777", "Airbus A380", "Boeing 747", "Airbus A320", "Boeing 737", "Embraer 190", "invalid", "invalid", "invalid", "invalid"]

fleet_neuralnetwork.Identify("0505_ao04.jpg", "fleet")


# Final save of the parameters
np.savez("neural-network-parameters.npz", self.parameters)
