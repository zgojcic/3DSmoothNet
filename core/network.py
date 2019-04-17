# network.py ---
#
# Filename: network.py
# Description: Network class, initializes the network
# Author: Zan Gojcic
#
# Project: 3DSmoothNet https://github.com/zgojcic/3DSmoothNet
# Created: 03.04.2019
# Version: 1.0

# Copyright (C)
# IGP @ ETHZ

# Code:

import numpy as np
import tensorflow as tf
import glob
import time
import os
import datetime
from core import ops

from tqdm import trange


class NetworkBuilder(object):
    """Network builder class """

    def __init__(self, config):

        self.config = config

        # Initialize tensorflow session
        self._init_tensorflow()

        # Build the network
        self._build_placeholder()
        self._build_data_loader()
        self._build_model()
        self._build_loss()
        self._build_optim()
        self._build_summary()
        self._build_writer()

    def _init_tensorflow(self):
        # Initialize tensorflow and let the gpu memory to grow
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=tf_config)

    def _build_placeholder(self):

        # Create placeholders for the input to the siamese network
        self.anchor_input = tf.placeholder(dtype=tf.float32, shape=[None, int(np.cbrt(self.config.input_dim)),
                                                                    int(np.cbrt(self.config.input_dim)),
                                                                    int(np.cbrt(self.config.input_dim)), 1],
                                           name='X_reference')

        self.positive_input = tf.placeholder(dtype=tf.float32, shape=[None, int(np.cbrt(self.config.input_dim)),
                                                                      int(np.cbrt(self.config.input_dim)),
                                                                      int(np.cbrt(self.config.input_dim)), 1],
                                             name='X_positive')

        # Global step for optimization
        self.global_step = tf.Variable(0, trainable=False)

    def _build_data_loader(self):

        if not os.path.exists(self.config.training_data_folder):
            print('Error directory: {}'.format(self.config.training_data_folder))
            raise ValueError('The training directory {} does not exist.'.format(self.config.training_data_folder))

            # Get name of all tfrecord files
        training_data_files = glob.glob(self.config.training_data_folder + '*.tfrecord')
        nr_training_files = len(training_data_files)
        print('Number of training files: {}'.format(nr_training_files))

        # Creates a data set that reads all of the examples from file names.
        dataset = tf.data.TFRecordDataset(training_data_files)

        # Parse the record into tensors.
        dataset = dataset.map(ops._parse_function)

        # Shuffle the data set
        dataset = dataset.shuffle(buffer_size=self.config.shuffle_size_TFRecords)

        # Repeat the input indefinitely
        dataset = dataset.repeat()

        # Generate batches
        dataset = dataset.batch(self.config.batch_size)
        dataset = dataset.prefetch(self.config.batch_size * 2)

        # Create a one-shot iterator
        iterator = dataset.make_one_shot_iterator()
        self.anc_training_batch, self.pos_training_batch = iterator.get_next()

    def _build_model(self):
        """Build 3DSmoothNet network for testing."""

        # -------------------- Network archintecture --------------------
        from core.architecture import network_architecture

        # Build graph
        print("Building the 3DSmoothNet graph")

        self.keep_probability = tf.placeholder(tf.float32)

        # Build network for training usinf the tf_records files
        self.anchor_output, self.positive_output = network_architecture(self.anc_training_batch,
                                                                        self.pos_training_batch,
                                                                        self.keep_probability, self.config)

        # Build network for testing and validation that uses the placeholders for data input
        self.test_anchor_output, self.test_positive_output = network_architecture(self.anchor_input,
                                                                                  self.positive_input,
                                                                                  self.keep_probability, self.config,
                                                                                  reuse=True)

    def _build_loss(self):
        # Import the loss function
        from core import loss

        # Create mask for the batch_hard loss
        positiveIDS = np.arange(self.config.batch_size)
        positiveIDS = tf.reshape(positiveIDS, [self.config.batch_size])

        self.dists = loss.cdist(self.anchor_output, self.positive_output, metric='euclidean')
        self.losses = loss.LOSS_CHOICES['batch_hard'](self.dists, positiveIDS,
                                                      self.config.margin, batch_precision_at_k=None)

        # tf.summary.scalar("loss", self.losses)

    def _build_optim(self):
        # Build the optimizer
        starter_learning_rate = self.config.learning_rate
        learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                   self.config.decay_step, self.config.decay_rate,
                                                   staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # Merge all summary op
        self.summary_op = tf.summary.merge_all()

        # Adam optimization, with the adaptable learning_rate
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.losses, global_step=self.global_step)
        self.optimization_parameters = [optimizer, self.losses, self.summary_op]

    def _build_summary(self):
        """Build summary ops."""

        # Merge all summary op
        self.summary = tf.summary.merge_all()

    def _build_writer(self):
        self.saver = tf.train.Saver()
        self.saver = tf.train.Saver(max_to_keep=self.config.max_epochs)

        self.time_stamp_format = "%f_%S_%H_%M_%d_%m_%Y"
        time_stamp = datetime.datetime.now().strftime(self.time_stamp_format)
        self.base_file_name = 'lr_{}_batchSize_{}_outDim_{}_{}'.format(self.config.learning_rate, self.config.batch_size,
                                                                       self.config.output_dim, time_stamp)


        # Initlaize writer for the tensorboard
        if not os.path.exists(self.config.log_path + '/{}_dim/'.format(self.config.output_dim)):
            os.makedirs(self.config.log_path + '/{}_dim/'.format(self.config.output_dim))
            print('Created a folder: {}'.format(self.config.log_path +
                                                '/{}_dim/'.format(self.config.output_dim)))

        # Check which saved files are already existing
        output_dir = os.listdir(self.config.log_path + '/{}_dim/'.format(self.config.output_dim))
        temp_names = [d.split('_') for d in output_dir]
        temp_names = list(map(int, [item[-1] for item in temp_names]))
        if len(temp_names) == 0:
            log_number = '0'
        else:
            log_number = str(np.max(temp_names) + 1)

        tensorboard_log = self.config.log_path + '/{}_dim/'.format(self.config.output_dim) + '/run_' + log_number
        self.writer = tf.summary.FileWriter(tensorboard_log, self.sess.graph)

    def train(self):

        # Initialize variables for accuracy values
        training_accuracy = []
        validation_accuracy = []

        # Load validation data
        if not os.path.exists(self.config.validation_data_folder):
            print('Error directory: {}'.format(self.config.validation_data_folder))
            raise ValueError('The validation data directory {} does not exist.'.format(self.config.validation_data_folder))

        validation_data_file = glob.glob(self.config.validation_data_folder + '*.npz')
        self.validation_data = np.load(validation_data_file[0])

        self.x_validate = self.validation_data['x']
        self.y_validate = self.validation_data['y']

        self.x_validate = np.reshape(self.x_validate, newshape=(-1, int(np.cbrt(self.config.input_dim)),
                                                                int(np.cbrt(self.config.input_dim)),
                                                                int(np.cbrt(self.config.input_dim)), 1))

        self.y_validate = np.reshape(self.y_validate, newshape=(-1, int(np.cbrt(self.config.input_dim)),
                                                                int(np.cbrt(self.config.input_dim)),
                                                                int(np.cbrt(self.config.input_dim)), 1))

        # Initialize all the variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # If resume load the trained model
        if self.config.resume_flag:

            print("Restoring the pretrianed model from {}".format(
                self.config.pretrained_model))

            if not os.path.exists(self.config.pretrained_model + '.index'):
                print('Error path: {}'.format(self.config.pretrained_model))
                raise ValueError('The pretrained model {} does not exist.'.format(self.config.pretrained_model))

            # Load the model (first run init as not all the variables are initialized when loaded)
            self.saver.restore(self.sess, self.config.pretrained_model)

            # Extract number of steps from the model name
            self.step = self.sess.run(self.global_step)
            print("Starting from the global step {}.".format(
            self.step))

        else:
            print("Starting from scratch!")
            self.step = 0

        for self.step in trange(self.step, self.config.max_steps, ncols=79):

            # If evaluateRate then check accuracy and save the model
            if (self.step + 1) % self.config.evaluate_rate == 0:
                embedded_anchor_train_features, embedded_positive_train_features = \
                    self.sess.run([self.anchor_output, self.positive_output],
                                  feed_dict={self.keep_probability: 1.0})[0:2]

                training_accuracy_temp = ops.compute_accuracy(embedded_anchor_train_features,
                                                              embedded_positive_train_features)

                print('\nOnline training accuracy at iterration {} equals {} percent!'.format(self.step,
                                                                                                training_accuracy_temp))
                # Perform a training and validation accuracy check
                training_accuracy.append(training_accuracy_temp)
                validation_accuracy.append(self.validation())

            # Save the model at the selected interval
            if (self.step + 1) % self.config.save_model_rate == 0:

                self.saver.save(self.sess,
                                save_path=self.config.saved_model_dir + '{}_dim/'.format(self.config.output_dim) +
                                self.base_file_name + '_trainedModel_Iteration_{}.ckpt'.format(self.step))

            # Save the mean accuracy to the tensorboard log at the selected interval
            if (self.step + 1) % self.config.save_accuracy_rate == 0:

                summary_training = tf.Summary(value=[tf.Summary.Value(tag='Training accuracy',
                                                                         simple_value=np.mean(training_accuracy))])
                summary_validation = tf.Summary(value=[tf.Summary.Value(tag='Validation accuracy',
                                                                         simple_value=np.mean(validation_accuracy))])

                self.writer.add_summary(summary_training, self.step)
                self.writer.add_summary(summary_validation, self.step)

                training_accuracy = []
                validation_accuracy = []


            # Training step
            _, loss_value, summary = self.sess.run(self.optimization_parameters,
                                                   feed_dict={self.keep_probability: self.config.dropout_rate})

            # Write data for tensorboard
            self.writer.add_summary(summary, self.step)
            self.step += 1


    def test(self):

        # Check if the selected model exists
        model_path = self.config.saved_model_dir + '{}_dim/'.format(self.config.output_dim)
        model_file_name = self.config.saved_model_evaluate + '_{}_dim.ckpt'.format(self.config.output_dim)

        if not os.path.exists(model_path + model_file_name + '.index'):
            print('Model File {} does not exist!'.format(model_path + model_file_name + '.index'))
            raise ValueError('The model {} does not exist.'.format(model_path + model_file_name + '.index'))

        # If model exists, load weights
        self.saver.restore(self.sess, model_path + model_file_name)
        print('Loaded saved model {0}.'.format(model_path + model_file_name))

        # Check if input data exists
        if not os.path.exists(self.config.evaluate_input_folder):
            print('Evaluate input data folder {} does not exist.'.format(self.config.evaluate_input_folder))
            raise ValueError('The input data folder {} does not exist.'.format(self.config.evaluate_input_folder))

        # Find all input files
        evaluation_files = glob.glob(self.config.evaluate_input_folder + '*.csv')

        for file in evaluation_files:
            print('Loading test file: ' + file)
            evaluation_features = np.fromfile(file, dtype=np.float32).reshape(-1, self.config.input_dim)

            # Reshape the feature so that they fit the input format
            evaluation_features = np.reshape(evaluation_features, newshape=(-1, int(np.cbrt(self.config.input_dim)),
                                                                            int(np.cbrt(self.config.input_dim)),
                                                                            int(np.cbrt(self.config.input_dim)), 1))

            # Generate batches for one epoch
            batches = ops.batch_iter(list(evaluation_features), self.config.evaluation_batch_size, 1, shuffle=False)
            all_predictions = []
            cnt = 0

            start = time.time()
            for x_test_batch in batches:

                batch_predictions = self.sess.run([self.test_anchor_output], feed_dict={self.anchor_input: x_test_batch,
                                                                                        self.keep_probability: 1.0})[0]
                if cnt == 0:
                    all_predictions = batch_predictions
                else:
                    all_predictions = np.concatenate([all_predictions, batch_predictions])
                cnt = cnt + 1

            end = time.time()

            print('{0} features computed in {1} seconds.'.format(len(evaluation_features), end - start))

            # Create output folder if it does not exist
            if not os.path.exists(self.config.evaluate_output_folder + '/{}_dim/'.format(self.config.output_dim)):
                os.makedirs(self.config.evaluate_output_folder + '/{}_dim/'.format(self.config.output_dim))
                print('Created a folder: {}'.format(self.config.evaluate_output_folder +
                                                    '/{}_dim/'.format(self.config.output_dim)))

            # Get the name of the file
            evaluate_file_name = file.split('/')[-1]

            # Save 3DSmoothNet descriptors as *.npz and *.txt
            np.savez_compressed(self.config.evaluate_output_folder + '/{}_dim/'.format(self.config.output_dim) +
                                evaluate_file_name[:-4] + '_3DSmoothNet.npz', data=all_predictions)

            np.savetxt(self.config.evaluate_output_folder + '/{}_dim/'.format(self.config.output_dim) +
                       evaluate_file_name[:-4] + '_3DSmoothNet.txt', all_predictions, delimiter=',', encoding=None)

            print('Wrote file {0}'.format(evaluate_file_name[:-4] + '_3DSmoothNet.npz'))


    def validation(self):
        # Use only one batch of validation point to do inference
        indices = np.random.choice(self.x_validate.shape[0], self.config.batch_size, replace=False)

        embedded_anchor_validation_features, embedded_positive_validation_features = self.sess.run(
            [self.test_anchor_output, self.test_positive_output],
            feed_dict={self.anchor_input: self.x_validate[indices], self.positive_input: self.y_validate[indices],
                       self.keep_probability: 1.0})[0:2]

        validation_accuracy_temp = ops.compute_accuracy(embedded_anchor_validation_features,
                                                        embedded_positive_validation_features)
        print('Online validation accuracy at iterration {} equals {} percent!'.format(self.step,
                                                                                        validation_accuracy_temp))

        return validation_accuracy_temp


#
# network.py ends here
