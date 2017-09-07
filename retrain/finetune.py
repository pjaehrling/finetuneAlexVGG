#
# Author: Philipp Jaehrling
# Influenced by: 
# - https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html
# - https://kratzert.github.io/2017/06/15/example-of-tensorflows-new-input-pipeline.html
#

import argparse

from models.alexnet import AlexNet
from models.vgg import VGG
from helper.imageloader import load_image_paths_by_subfolder
from helper.retrainer import Retrainer

from tensorflow.python.platform import gfile

# Input params
VALIDATION_RATIO = 5 # every 5th element = 1/5 = 0.2 = 20%

# Learning params
LEARNING_RATE = 0.01 
NUM_EPOCHS = 2
BATCH_SIZE = 128

# Network params
KEEP_PROB = 1.0 # [0.5]
FINETUNE_LAYERS = ['fc7', 'fc8']


def finetune(root_dir, ckpt, model_def):
    """
    Args:
        root_dir:
        ckpt:
        model_def:
    """
    image_paths = load_image_paths_by_subfolder(root_dir, VALIDATION_RATIO)
    if image_paths['training_image_count'] < BATCH_SIZE:
        print 'Not enough training images in \'%s\'' %root_dir
        return
    
    trainer = Retrainer(model_def, image_paths)
    trainer.run(
        FINETUNE_LAYERS,
        NUM_EPOCHS,
        LEARNING_RATE,
        BATCH_SIZE,
        KEEP_PROB,
        ckpt
    )


def main():
    """
    Main
    """
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'root_dir',
        help='Folder with trainings/validation images'
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
    root_dir = args.root_dir
    ckpt = args.ckpt
    model_str = args.model

    if not gfile.Exists(root_dir):
        print 'Image root directory \'%s\' not found' %root_dir
        return None

    if ckpt and not gfile.Exists(ckpt):
        print 'Could not find checkpoint file: \'%s\'' %ckpt
        return None

    if model_str == 'vgg':
        model_def = VGG
    elif model_str == 'alex': # default
        model_def = AlexNet

    finetune(root_dir, ckpt, model_def)

if __name__ == '__main__':
    main()