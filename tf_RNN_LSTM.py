import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

# Define parameters
# The max number of words to use, the most frequent ones
number_of_words = 20000
# The max length of the word vectors, we use padding when required
max_len = 100
# The size of the vocabulary
vocab_size = number_of_words
# The embedding size, the size of the embedding vectors
embed_size = 120

# Loading the data
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=number_of_words)

# Padding all sequences so that they have all the same length
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

"""
Defining the Recurrent Neural Network
"""
model = Sequential()

# Adding the embedding layer
model.add(Embedding(vocab_size,
                    embed_size,
                    input_shape=(X_train.shape[1],)))

# Adding the LSTM layer
# units: 128
# activation: tanh
model.add(LSTM(units=128, activation='tanh'))

model.add(Dropout(0.3))

# Adding the Dense output layer
# units: 1
# activation: sigmoid
model.add(Dense(units=1, activation='sigmoid'))

# Compiling the model
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Printing the summary of the model
print(model.summary())

"""
Training the model
"""
model.fit(X_train, y_train, epochs=6, batch_size=128)

"""
Evaluating the model
"""
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Test accuracy: {}".format(test_accuracy))
