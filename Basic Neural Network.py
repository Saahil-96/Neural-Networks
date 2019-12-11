import numpy as np
import random


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    return x*(1-x)


training_inputs = np.array(([0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]))
training_outputs = np.array(([0], [1], [1], [0]))

Arr=np.array(([0], [0], [0]))
Cost_Arr=np.array(([0]))

weights = np.array(([random.random()], [random.random()], [random.random()]))

for j in range(0, 20000):

    inputs=training_inputs
    Neuron = sigmoid(np.dot(inputs,weights))
    Cost = (training_outputs-Neuron)
    adjustments=Cost*sigmoid_der(Neuron)
    weights+=np.dot(inputs.T, adjustments)
    Cost_Arr = np.append(Cost_Arr, Cost)
    Arr=np.append(Arr, weights, 1)

Cost_Arr=np.delete(Cost_Arr, 0)
Arr=np.delete(Arr, 0, 1)

print(Cost_Arr)
print(np.amin(Cost_Arr))
#x=np.where(Cost_Arr==Cost_Arr.min())
#print(x)
#print(Arr[:,x])
#Out=sigmoid(Arr[0,x]*1+Arr[1,x]*0+Arr[2,x]*0)
print(Neuron)