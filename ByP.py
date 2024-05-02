
# Basic import of TensorFlow

import tensorflow as tf
from tensorflow import keras
# Keras is a powerful and user-friendly deep learning API written in Python.
import numpy as np
# numpy is a package used for mathematical calculation and array manuplation

# Declaration of constant
msg = tf.constant('Hello, TensorFlow!')
tf.print(msg)


# The MNIST database (Modified National Institute of Standards and Technology database)
# is a large database of handwritten digits that is commonly used for training various image processing systems.

mnist = tf.keras.datasets.mnist
Sampdat = tf.keras.datasets.imdb

# Define data set
# Creation of models/ simple neural network
# Keras is a neural network library
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

#Units -
#Input_Shape -

# Compling the model with optimizer and loss

model.compile(optimizer='sgd', loss='mean_squared_error')

# Define sample data

# xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
# ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

xs = np.array([-1,  0, 1, 2, 3, 4], dtype=int)
ys = np.array([-3, -1, 1, 3, 5, 7], dtype=int)


# Training the model

model.fit(xs, ys, epochs=50)

# Now check if it works

guess = model.predict([1])
tf.print('MyGuess  ', guess)

msg = tf.constant('program ended')
tf.print(msg)