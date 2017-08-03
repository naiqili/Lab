import tensorflow as tf

def activation(x):
    #return 1.7159 * tf.tanh(0.6666 * x)
    return tf.tanh(x)

def sigmoid(x):
    #return tf.nn.relu(x)
    return tf.sigmoid(x)