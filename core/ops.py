# ops.py ---
#
# Filename: ops.py
# Description: Contains helper function for running the 3DSmoothNet code
# Author:  Zan Gojcic, Caifa Zhou
# Project: 3DSmoothNet https://github.com/zgojcic/3DSmoothNet
# Created: 03.04.2019
# Version: 1.0

# Code:

# Import python dependencies
import tensorflow as tf
import numpy as np
from sklearn.neighbors import NearestNeighbors


# Create lots of weights and biases & Initialize with a small positive number as we will use ReLU
def weight(shape, layer_name, weight_initializer=None,reuse=False):
    if weight_initializer is None:
        weight_initializer = tf.orthogonal_initializer(gain=0.6,
                                                       seed=41,
                                                       dtype=tf.float32)
    with tf.name_scope(layer_name):
        with tf.variable_scope(layer_name, reuse=reuse):
            weights = tf.get_variable(layer_name + "_W", shape=shape,
                                      dtype=tf.float32, initializer=weight_initializer)

    tf.summary.histogram(layer_name, weights)
    return weights


def bias(shape, layer_name,reuse=False):
    bias_init = tf.constant_initializer(0.01)

    with tf.name_scope(layer_name):
        with tf.variable_scope('', reuse=reuse):
            biases = tf.get_variable(layer_name + '_b',  shape=shape,
                                     dtype=tf.float32, initializer=bias_init)  # default initialier: glorot_uniform_initializer
    return biases


def conv3d(x,filtertype , stride, padding):
    return tf.nn.conv3d(x, filter=filtertype, strides=[1, stride[0], stride[1], stride[2], 1], padding=padding)


def max_pool3d(x, kernel, stride, padding):
    return tf.nn.max_pool3d(x, ksize=kernel, strides=stride, padding=padding)


def avg_pool3d(x, kernel, stride, padding):
    return tf.nn.avg_pool3d(x, ksize=kernel, strides=stride, padding=padding)


def relu(x):
    return tf.nn.relu(x)


def batch_norm(x):
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(x, [0])
    return tf.nn.batch_normalization(x, batch_mean, batch_var, None, None, epsilon)


def l2_normalize(x):
    return tf.nn.l2_normalize(x, axis=1, epsilon=1e-12, name=None)


def dropout(x,dropout_rate=0.7):
    return tf.nn.dropout(x,keep_prob=dropout_rate,noise_shape=None,seed=None,name=None)


def compute_accuracy(embeddedRefFeatures, embeddedValFeatures):
    numberOfTestPoints = embeddedRefFeatures.shape[0]
    neigh = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', metric='euclidean')
    neigh.fit(embeddedRefFeatures)
    distNeighNormal, indNeighNormal = neigh.kneighbors(embeddedValFeatures)
    referenceNeighbors = np.reshape(np.arange(numberOfTestPoints), newshape=(-1, 1))

    wrongMatches = np.count_nonzero(indNeighNormal - referenceNeighbors)
    accuracy = (1 - wrongMatches / numberOfTestPoints) * 100
    return accuracy


def flatten_list(l):
    flat_list = [item for sublist in l for item in sublist]
    return flat_list


def _parse_function(example_proto):
    inputFormat = (16,16,16,1)
    keys_to_features = {'X': tf.FixedLenFeature(inputFormat, tf.float32),
                        'Y': tf.FixedLenFeature(inputFormat, tf.float32)}
    parsed_features = tf.parse_single_example(example_proto, keys_to_features)

    return parsed_features['X'], parsed_features['Y']


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


#
# ops.py ends here