import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print(train_labels)

#open('Stop-Sign-Detection-master/Stop-Sign-Detection-master/Stop Sign Dataset/via_region_data.json').read()
class_names=['T-shirt/top',
             'Trouser',
             'Pullover',
             'Dress',
             'Coat',
             'Sandal',
             'Shirt',
             'Sneaker',
             'Bag',
             'Ankle boot']

# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

train_images=train_images/255.0 #Regularization
test_images=test_images/255.0

#Construct sequential series of layers

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), # Flatten Layer to a 1D column array
    keras.layers.Dense(128, activation=tf.nn.relu), # Make a 128 layer node that is densely connected to previous layer
    keras.layers.Dense(10, activation=tf.nn.softmax) # Returns 10 probability scores; Softmax regression
    ])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy", # Cost function
              metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=10) # More the number of epochs, data gets overfitted. Less number of epochs, data gets underfitted

test_loss, test_acc = model.evaluate(test_images, test_labels)

print(f"Test Accuracy: {test_acc}")
predictions = model.predict(test_images)

print(predictions[10])
print(np.argmax(predictions[10]))
print(test_labels[10])

plt.figure()
plt.imshow(test_images[10])
plt.colorbar()
plt.grid(False)
plt.show()
