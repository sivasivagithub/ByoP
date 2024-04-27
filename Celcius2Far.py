import tensorflow as tf
# Importing tensor flow the platform API
import numpy as np
# Importing numpy library which is use for mathematical operations and manipulating arrays
import logging

# Logging library in python
logger = tf.get_logger()
# get_logger is part of the logging module- Python documentation should be researched
logger.setLevel(logging.ERROR)

# In this case celcius is the feature
celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)
# In this case farenhit is the out or label

for i, c in enumerate(celsius_q):
    print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))

# Enumrate is used for iteration and it is part of the Python library it fixes the index starting with 0
# and items are taken one by one for computing
# for i,c in enumerate(celsius_q):
#   print("{} c {} i ".format(c, i))
# for i, c in enumerate(celsius_q):
#        print(c,i)
## Create the model
# Next, create the model. We will use the simplest possible model we can, a Dense network.
# Since the problem is straightforward, this network will require only a single layer, with a single neuron.
### Build a layer
# We'll call the layer `l0` and create it by instantiating `tf.keras.layers.Dense` with the following configuration:
# `input_shape=[1]` — This specifies that the input to this layer is a single value.
# That is, the shape is a one-dimensional array with one member. Since this is the first (and only) layer,
# that input shape is the input shape of the entire model.
# The single value is a floating point number, representing degrees Celsius.
# units=1` — This specifies the number of neurons in the layer. The number of neurons defines how many internal variables
# the layer has to try to learn how to solve the problem (more later). Since this is the final layer,
# it is also the size of the model's output — a single float value representing degrees Fahrenheit.
# (In a multi-layered network, the size and shape of the layer would need to match the `input_shape` of the next layer.)

l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
