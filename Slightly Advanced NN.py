import numpy as np

class NeuralNetwork:
    def _init_(self):
        np.random.seed(1)

    def synaptic_weights(self):
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1
        return self.synaptic_weights

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_der(self, x):
        return x * (1 - x)


    def train(self, training_inputs, training_outputs, training_iterations):
        for iteration in range(training_iterations):
            output = self.think(training_inputs)
            error=training_outputs - output
            adjustments = np.dot(training_inputs.T, error*self.sigmoid_der(output))
            self.synaptic_weights += adjustments

    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))

        return output

neural_network = NeuralNetwork()

print("Random Weights: ")
n=neural_network.synaptic_weights()
print(n)

training_inputs = np.array(([0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]))
training_outputs = np.array(([0], [1], [1], [0]))
neural_network.train(training_inputs, training_outputs, 60000)
print(neural_network.synaptic_weights)

A = str(input("Input 1:"))
B = str(input("Input 2:"))
C = str(input("Input 3:"))

print("New Situation: input data=", A, B, C)
print("Output:")
print(neural_network.think(np.array([A, B, C])))
