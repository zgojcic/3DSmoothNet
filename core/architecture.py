# architecture.py ---
#
# Filename: architecture.py
# Description: defines the architecture of the 3DSmoothNet
# Author: Zan Gojcic, Caifa Zhou
#
# Project: 3DSmoothNet https://github.com/zgojcic/3DSmoothNet
# Created: 04.04.2019
# Version: 1.0

# Copyright (C)
# IGP @ ETHZ

# Code:

# Import python dependencies
import tensorflow as tf
import numpy as np

# Import custom functions
from core import ops


def network_architecture(x_anc,x_pos, dropout_rate, config, reuse=False):

    # Join the 3DSmoothNet structure with the desired output dimension
    net_structure = [1, 32, 32, 64, 64, 128, 128]
    outputDim = config.output_dim
    channels = [item for sublist in [net_structure, [outputDim]] for item in sublist]

    # In the third layer stride is 2
    stride = np.ones(len(channels))
    stride[2] = 2

    # Apply dropout in the 6th layer
    dropout_flag = np.zeros(len(channels))
    dropout_flag[5] = 1

    # Initalize data
    input_anc = x_anc
    input_pos = x_pos
    layer_index = 0

    # Loop over the desired layers
    with tf.name_scope('3DIM_cnn') as scope:
        for layer in np.arange(0, len(channels)-2):
            scope_name = "3DIM_cnn" + str(layer_index+1)
            with tf.name_scope(scope_name) as inner_scope:
                input_anc, input_pos = conv_block(input_anc, input_pos, [channels[layer], channels[layer + 1]],
                                                  dropout_flag[layer], dropout_rate, layer_index,
                                                  stride_input=stride[layer], reuse=reuse)

            layer_index += 1

        with tf.name_scope('3DIM_cnn7') as inner_scope:
            input_anc, input_pos = out_block(input_anc, input_pos, [channels[-2], channels[-1]],
                                             layer_index, reuse=reuse)

        return ops.l2_normalize(input_anc), \
               ops.l2_normalize(input_pos)



def conv_block(input_anc, input_pos, channels, dropout_flag, dropout_rate, laxer_idx, stride_input=1, k_size=3,
               padding_type='SAME', reuse=False):

    # Traditional 3D conv layer followed by batch norm and relu activation

    i_size = input_anc.get_shape().as_list()[-2]/stride_input

    weights = ops.weight([k_size, k_size, k_size, channels[0], channels[1]],
                         layer_name='wcnn' + str(laxer_idx+1), reuse=reuse)

    bias = ops.bias([i_size, i_size, i_size, channels[1]], layer_name='bcnn' + str(laxer_idx+1),reuse=reuse)

    conv_output_anc = tf.add(ops.conv3d(input_anc, weights, stride=[stride_input,stride_input, stride_input], padding=padding_type),bias)
    conv_output_pos = tf.add(ops.conv3d(input_pos, weights, stride=[stride_input, stride_input, stride_input], padding=padding_type),bias)

    conv_output_anc = ops.batch_norm(conv_output_anc)
    conv_output_pos = ops.batch_norm(conv_output_pos)

    conv_output_anc = ops.relu(conv_output_anc)
    conv_output_pos = ops.relu(conv_output_pos)

    if dropout_flag:
        conv_output_anc = ops.dropout(conv_output_anc, dropout_rate=dropout_rate)
        conv_output_pos = ops.dropout(conv_output_pos, dropout_rate=dropout_rate)

    return conv_output_anc, conv_output_pos

def out_block(input_anc, input_pos, channels, laxer_idx, stride_input=1, k_size=8, padding_type = 'VALID', reuse=False):

    # Last conv layer, flatten the output
    weights = ops.weight([k_size, k_size, k_size, channels[0], channels[1]],
                         layer_name='wcnn' + str(laxer_idx+1), reuse=reuse)

    bias = ops.bias([1, 1, 1, channels[1]], layer_name='bcnn' + str(laxer_idx + 1),reuse=reuse)

    conv_output_anc = tf.add(ops.conv3d(input_anc, weights, stride=[stride_input,stride_input, stride_input], padding=padding_type), bias)
    conv_output_pos = tf.add(ops.conv3d(input_pos, weights, stride=[stride_input, stride_input, stride_input], padding=padding_type), bias)

    conv_output_anc = ops.batch_norm(conv_output_anc)
    conv_output_pos = ops.batch_norm(conv_output_pos)

    conv_output_anc = tf.contrib.layers.flatten(conv_output_anc)
    conv_output_pos = tf.contrib.layers.flatten(conv_output_pos)

    return conv_output_anc, conv_output_pos


