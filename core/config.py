# config.py ---
#
# Filename: config.py
# Description: Based on config file from https://github.com/vcg-uvic/learned-correspondence-release
# Author:  Zan Gojcic, Caifa Zhou
# Project: 3DSmoothNet https://github.com/zgojcic/3DSmoothNet 
# Created: 03.04.2019
# Version: 1.0	

# Code:

# Import python dependencies
import argparse


arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# -----------------------------------------------------------------------------
# Network
net_arg = add_argument_group("Network")
net_arg.add_argument("--run_mode", type=str, default="test",
                     help='run_mode')
net_arg.add_argument('--input_dim', type=int, default=4096,
                     help='the dimension of the input features')
net_arg.add_argument('--output_dim', type=int, default=32,
                     help='the dimension of the learned local descriptor')
net_arg.add_argument('--log_path', type=str, default='./logs',
                     help='path to the directory with the tensorboard logs')
# -----------------------------------------------------------------------------
# Test
test_arg = add_argument_group("Evaluate")
test_arg.add_argument("--evaluate_input_folder", type=str, default="./data/evaluate/input_data/",
                          help='prefix for the input folder locations')
test_arg.add_argument("--evaluate_output_folder", type=str, default="./data/evaluate/output_data/",
                          help='prefix for the output folder locations')
test_arg.add_argument('--evaluation_batch_size', type=int, default=1000,
                          help='the number of examples for each iteration of inference')
test_arg.add_argument('--saved_model_dir', type=str, default='./models/',
                     help='the directory of the pre-trained model')
test_arg.add_argument('--saved_model_evaluate', type=str, default='3DSmoothNet',
                     help='file name of the model to load')
# -----------------------------------------------------------------------------
# Training
train_arg = add_argument_group("Train")
train_arg.add_argument("--input_data_folder", type=str, default="./data/train/input_data/",
                      help='prefix for the input folder locations')
train_arg.add_argument("--output_data_folder", type=str, default="./data/train/output_data/",
                      help='prefix for the output folder locations')
train_arg.add_argument('--max_steps', type=int, default=20000000,
                       help='maximum number of training iterations')
train_arg.add_argument('--max_epochs', type=int, default=20,
                       help='maximum number of training epochs')
train_arg.add_argument('--batch_size', type=int, default=256,
                       help='the number of training examples for each iteration')
train_arg.add_argument('--learning_rate', type=float, default=1e-3,
                       help='the initial learning rate')
train_arg.add_argument('--evaluate_rate', type=int, default=100,
                       help='frequency of evaluation')
train_arg.add_argument('--save_model_rate', type=int, default=1000,
                       help='the frequency of saving the check point')
train_arg.add_argument('--save_accuracy_rate', type=int, default=500,
                       help='the frequency of saving the training and validation accuracy')
train_arg.add_argument('--margin', type=str, default='soft',
                       help='the margin fucntion used for the loss')
train_arg.add_argument('--dropout_rate', type=float, default=0.7,
                       help='the keep probability')
train_arg.add_argument('--resume_flag', type=int, default=0,
                       help='the flag for training using the pre-trained model (1) or not (0)')
train_arg.add_argument('--decay_rate', type=float, default=0.95,
                       help='the rate of exponential learning rate decaying')
train_arg.add_argument('--decay_step', type=int, default=5000,
                       help='the frequency of exponential learning rate decaying')
train_arg.add_argument('--shuffle_size_TFRecords', type=int, default=5000,
                       help='the shuffle buffer size of the TFRecords')
train_arg.add_argument('--training_data_folder', type=str, default="./data/train/trainingData3DMatch",
                       help='location of the training data files')
train_arg.add_argument('--pretrained_model', type=str, default="./models/32_dim/3DSmoothNet_32_dim.ckpt",
                       help='pretrained model which will be used if resume is activared')

# Validation
valid_arg = add_argument_group("Validation")
train_arg.add_argument('--validation_data_folder', type=str, default="./data/validation/validationData3Dmatch/",
                       help='location of the validation data files')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed

def print_usage():
    parser.print_usage()

#
# config.py ends here
