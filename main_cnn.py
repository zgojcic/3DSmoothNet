# main.py ---
#
# Filename: config.py
# Description: Main file of the 3DSmoothNet feature descriptor. Parameters can be seen in the core/config.py file.
#
# Author: Gojcic Zan, Zhou Caifa
# Project: 3DSmoothNet https://github.com/zgojcic/3DSmoothNet
# Paper: https://arxiv.org/abs/1811.06879
# Created: 03.04.2019
# Version: 1.0

# Copyright (C)
# IGP @ ETHZ

# Code:

# Import python dependencies
import tensorflow as tf

# Import custom functions
from core import config
from core import network


print('The version of TF is {}'.format(tf.__version__))

config_arguments, unparsed_arguments = config.get_config()

def main(config_arguments):

    # Build the model and optimizer
    smooth_net = network.NetworkBuilder(config_arguments)

    print('Run mode "{}" selected.'.format(config_arguments.run_mode))
    # Select the run mode
    if config_arguments.run_mode == "train":
        smooth_net.train()
    elif config_arguments.run_mode == "test":

        # Evaluate the network
        smooth_net.test()

    else:
        raise ValueError('%s is not a valid run mode.'.format(config_arguments.run_mode))



if __name__ == "__main__":

    # Parse configuration
    config_arguments, unparsed_arguments = config.get_config()

    # If we have unparsed arguments, print usage and exit
    if len(unparsed_arguments) > 0:
        config.print_usage()
        exit(1)

    main(config_arguments)

#
# main.py ends here
