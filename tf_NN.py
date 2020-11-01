"""
Author: Jesus A. Gonzalez
Date: 10/31/2020
Artificial Neural Network in Tensorflow 2.0
Classification for the MNIST Fashion dataset - Kaggle
"""

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Dense, Dropout

# Load the dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

"""
Data Preprocessing
"""

# Normalize the images for better results working with the NN
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshaping the data. This is not a CNN, then we need to input the data
# as 1D vectors. The MNIST images have a resolution of 28x28 pixels.
X_train = X_train.reshape(-1, 28*28)
print(X_train.shape)
X_test = X_test.reshape(-1, 28*28)
print(X_test.shape)


"""
ANN Definition
"""
# Defining the model object
model = tf.keras.models.Sequential()

# Adding the input layer
model.add(Dense(units=128, activation='relu', input_shape=(784, )))
# Adding hidden layers: dense and dropout
model.add(Dropout(0.2))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.4))
#model.add(Dense(units=128, activation='relu'))
#model.add(Dropout(0.4))
model.add(Dense(units=10, activation='softmax'))

# Compiling the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics='sparse_categorical_accuracy')

# Print the model summary
model.summary()

"""
Fitting the model
"""
model.fit(X_train, y_train, epochs=20)

"""
Evaluating the model
"""
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test accuracy: {}".format(test_accuracy))

"""
Saving the model
"""
# Saving the NN architecture - topology
model_json = model.to_json()
with open("fashion_model.json", "w") as json_file:
    json_file.write(model_json)
# Saving the NN weights
model.save_weights("fashion_model.h5")

