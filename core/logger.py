# config.py ---
#
# Filename: config.py
# Description: Based on argparse usage from
#              https://github.com/carpedm20/DCGAN-tensorflow
# Author: Zhou Caifa
# Project: 3DSmoothNet https://github.com/zgojcic/3DSmoothNet
# Created: 03.04.2019
# Version: 1.0

# Copyright (C)
# IGP @ ETHZ

# Code:

import logging
# import auxiliary_module


def loggerGenerator(name=None, level=logging.DEBUG, fileName='log.log'):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # create logger with 'spam_application'
    logger = logging.getLogger(name=name)
    logger.setLevel(level=level)


    # create file handler which logs even debug messages
    fh = logging.FileHandler(fileName)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    ch.setFormatter(formatter)

    # add to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger