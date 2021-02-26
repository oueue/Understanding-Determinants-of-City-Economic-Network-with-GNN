import tensorflow as tf
import numpy as np

def glorot(tensor, name=None):
    shape = list(tensor.size())
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def zeros(tensor, name = None):
    if tensor is not None:
        tensor.data.fill_(0)

