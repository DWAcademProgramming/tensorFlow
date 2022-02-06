#Import Statements
import tensorflow as tf
from tensorflow.keras.datasets import imdb

#Setting the variables
number_of_words = 20000
max_len = 100

#Training the dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=number_of_words)
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_len)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_len)

#Building the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=number_of_words, output_dim=128, input_shape=(X_train.shape[1],)))
model.add(tf.keras.layers.LSTM(units=128, activation='tanh'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#Displaying the model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
