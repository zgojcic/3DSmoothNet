import numpy as np
import os
import pickle
from os.path import isfile, join, sep
import subprocess
from glob import glob
import tensorflow as tf


def npy_to_tfrecords(features,pairs,output_file):
    # write records to a tfrecords file
    writer = tf.python_io.TFRecordWriter(output_file)

    # Loop through all the features you want to write
    for index in range(0, pairs.shape[0]):
        # Feature contains a map of string to feature proto objects
        feature = {}
        feature['X'] = tf.train.Feature(float_list=tf.train.FloatList(value=features[pairs[index,0],:]))
        feature['Y'] = tf.train.Feature(float_list=tf.train.FloatList(value=features[pairs[index,1],:]))

        # Construct the Example proto object
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize the example to a string
        serialized = example.SerializeToString()

        # write the serialized objec to the disk
        writer.write(serialized)


    writer.close()





directory = sys.argv[1]
string = directory.split('/')
featuresFolder = '03_Features/'
indicesFolder = 'Correspondences/'

tempPairs = np.load(directory + indicesFolder + 'mapped_indices.npz')
pairs = tempPairs['mapped_indices'].astype(int)

tempFeatures = np.load(directory + featuresFolder + '_Training.npz')
features = tempFeatures['data']

output_file = '/cluster/scratch/zgojcic/createTrainingData/completed/trainingData/' + 'file_{}.tfrecord'.format(string[5])

npy_to_tfrecords(features,pairs,output_file)

