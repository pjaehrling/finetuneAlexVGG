#
# Author: Philipp Jaehrling
# Influenced by: 
# - https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html
# - https://kratzert.github.io/2017/06/15/example-of-tensorflows-new-input-pipeline.html
#

import argparse
import tensorflow as tf

from models.alexnet import AlexNet
from models.vgg import VGG
from models.vgg_slim import VGGslim
from models.inception_v3 import InceptionV3
from helper.imageloader import load_image_paths_by_subfolder, load_image_paths_by_file
from helper.retrainer import Retrainer

from tensorflow.python.platform import gfile

# Input params
VALIDATION_RATIO = 10 # every 5th element = 1/5 = 0.2 = 20%
USE_SUBFOLDER = True
SKIP_FOLDER = ['yiwen']

# Learning params
LEARNING_RATE = 0.005
# TODO: try learning rate decay
# see: https://www.tensorflow.org/versions/r0.12/api_docs/python/train/decaying_the_learning_rate
NUM_EPOCHS = 20
BATCH_SIZE = 32

# Network params
KEEP_PROB = 1.0 # [0.5]
FINETUNE_LAYERS = ['fc6', 'fc7', 'fc8']

# HARDWARE USAGE
DEVICE = '/cpu:0'
MEMORY_USAGE = 1.0


def finetune(image_paths, ckpt, model_def):
    """
    Args:
        image_paths:
        ckpt:
        model_def:
    """
    trainer = Retrainer(model_def, image_paths)
    trainer.run(
        FINETUNE_LAYERS,
        NUM_EPOCHS,
        LEARNING_RATE,
        BATCH_SIZE,
        KEEP_PROB,
        MEMORY_USAGE,
        DEVICE,
        ckpt
    )

def main():
    """
    Main
    """
    # Make sure the logging output is visible.
    tf.logging.set_verbosity(tf.logging.INFO)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-image_dir',
        type=str,
        default='',
        help='Folder with trainings/validation images'
    )
    parser.add_argument(
        '-image_file',
        type=str,
        default='',
        help='File with a list of trainings/validation images and their labels'
    )
    parser.add_argument(
        '-ckpt',
        type=str,
        default='',
        help='Load this checkpoint file to continue training from this point on'
    )
    parser.add_argument(
        '-model',
        type=str,
        default='alex',
        help='Model to be validated'
    )
    
    args = parser.parse_args()
    image_dir = args.image_dir
    image_file = args.image_file
    ckpt = args.ckpt
    model_str = args.model

    # Load images
    if not image_dir and not image_file:
        print('Provide one of the following options to load images \'-image_file\' or \'-image_path\'' %image_dir)
        return None
    elif image_dir: 
        if not gfile.Exists(image_dir):
            print('Image root directory \'%s\' not found' %image_dir)
            return None
        else:
            image_paths = load_image_paths_by_subfolder(image_dir, VALIDATION_RATIO, SKIP_FOLDER, use_subfolder=USE_SUBFOLDER)
    else:
        if not gfile.Exists(image_file):
            print('Image file \'%s\' not found' %image_file)
            return None
        else:
            image_paths = load_image_paths_by_file(image_file, VALIDATION_RATIO)

    # Make sure we have enough images to fill at least one training/validation batch
    if image_paths['training_image_count'] < BATCH_SIZE:
        print('Not enough training images in \'%s\'' %image_dir)
        return None

    if image_paths['validation_image_count'] < BATCH_SIZE:
        print('Not enough validation images in \'%s\'' %image_dir)
        return None

    # Set a CNN model definition
    if model_str == 'vgg':
        model_def = VGG
    elif model_str == 'vgg_slim':
        model_def = VGGslim
    elif model_str == 'inc_v3':
        model_def = InceptionV3
    elif model_str == 'alex': # default
        model_def = AlexNet

    # Start retraining/finetuning
    finetune(image_paths, ckpt, model_def)

if __name__ == '__main__':
    main()