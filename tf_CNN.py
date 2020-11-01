"""
Author: Jesus A. Gonzalez
Date: 10/31/2020
Convolutional Neural Network in Tensorflow 2.0
Classification for the CIFAR10 dataset
These are color images, then there are 3 color channels
The resolution of the images is 32x32
"""
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout
import matplotlib.pyplot as plt

# Loading the data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Defining the classes names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Normalizing the data
X_train = X_train / 255.0
print(X_train.shape)
X_test =  X_test / 255.0
print(X_test.shape)

# Plotting an image
plt.imshow(X_train[7])

"""
Defining the model
"""
model = Sequential()

# Adding the first layer
# filters: 32
# kernel_size: 3
# padding: same
# activation: relu
# input_shape: (32, 32, 3)
model.add(Conv2D(filters=32,
                 kernel_size=3,
                 padding='same',
                 activation='relu',
                 input_shape=[32, 32, 3]))

# Adding the second layer
# filters: 32
# kernel_size: 3
# padding: same
# activation: relu
model.add(Conv2D(filters=32,
                 kernel_size=3,
                 padding='same',
                 activation='relu'))

# Adding a pooling layer
# pool_size: 2
# strides: 2
# padding: valid
model.add(MaxPool2D(pool_size=2,
                    strides=2,
                    padding='valid'))

# Adding the third layer
# filters: 64
# kernel_size: 3
# padding: same
# activation: relu
model.add(Conv2D(filters=64,
                 kernel_size=3,
                 padding='same',
                 activation='relu'))

# Adding the fourth layer
# filters: 64
# kernel_size: 3
# padding: same
# activation: relu
model.add(Conv2D(filters=64,
                 kernel_size=3,
                 padding='same',
                 activation='relu'))

# Adding a pooling layer
# pool_size: 2
# strides: 2
# padding: valid
model.add(MaxPool2D(pool_size=2,
                    strides=2,
                    padding='valid'))

# Adding a flatten layer
model.add(Flatten())

# Adding a dropout layer
model.add(Dropout(0.3))

# Adding a dense layer
# units: 128
# activation: relu
model.add(Dense(units=128, activation='relu'))

# Adding a dropout layer
model.add(Dropout(0.3))

# Adding a second dense layer
# units: 10
# activation: softmax
model.add(Dense(units=10, activation='softmax'))

# Printing the summary of the model
print(model.summary())

# Compiling the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

"""
Training the model
"""
model.fit(X_train, y_train, epochs=10, batch_size=512)

"""
Testing the model
"""
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Test accuracy: {}".format(test_accuracy))
