#
# Author: Philipp Jaehrling
#
import os
import argparse
import helper.data_provider as data_provider

from models.alexnet import AlexNet
from models.vgg import VGG
from models.vgg_slim import VGGslim
from models.inception_v3 import InceptionV3
from helper.retrainer import Retrainer

# Input params
VALIDATION_RATIO = 5 # e.g. 5 -> every 5th element = 1/5 = 0.2 = 20%
USE_SUBFOLDER = True
SKIP_FOLDER = ['yiwen']

# Learning params
LEARNING_RATE = 0.005
# TODO: try manual learning rate decay (if adam opt isn't used)
# see: https://www.tensorflow.org/versions/r0.12/api_docs/python/train/decaying_the_learning_rate
NUM_EPOCHS = 10
BATCH_SIZE = 32

# Network params
KEEP_PROB = 1.0 # [0.5]
FINETUNE_LAYERS = ['fc6', 'fc7', 'fc8']
CHECKPOINT_DIR = '../checkpoints'

# HARDWARE USAGE
DEVICE = '/gpu:0'
MEMORY_USAGE = 1.0

def finetune(model_def, data, ckpt_dir, write_checkpoint_on_each_epoch, init_from_ckpt, use_adam_optimizer):
    """
    Args:
        model_def:
        data:
        show_misclassified:
        validate_on_each_epoch:
        ckpt_dir:
        write_checkpoint_on_each_epoch:
        init_from_ckpt:
    """
    trainer = Retrainer(model_def, data, write_checkpoint_on_each_epoch)
    trainer.run(
        FINETUNE_LAYERS,
        NUM_EPOCHS,
        LEARNING_RATE,
        BATCH_SIZE,
        KEEP_PROB,
        MEMORY_USAGE,
        DEVICE,
        ckpt_dir,
        init_from_ckpt,
        use_adam_optimizer
    )

def main():
    """
    Main
    """
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
        '-write_checkpoint_on_each_epoch',
        default=False,
        help='Write a checkpoint file on each epoch (default is just once at the end',
        action='store_true' # whenever this option is set, the arg is set to true
    )
    parser.add_argument(
        '-init_from_ckpt',
        type=str,
        default='',
        help='Load this checkpoint file to continue training from this point on'
    )
    parser.add_argument(
        '-model',
        type=str,
        choices=['vgg', 'alex'],
        default='alex',
        help='Model to be validated. Default is AlexNet (alex)'
    )
    parser.add_argument(
        '-use_adam_optimizer',
        default=False,
        help='Use Adam optimizer instead of GradientDecent',
        action='store_true' # whenever this option is set, the arg is set to true
    )
    
    args = parser.parse_args()
    image_dir = args.image_dir
    image_file = args.image_file
    write_checkpoint_on_each_epoch = args.write_checkpoint_on_each_epoch
    init_from_ckpt = args.init_from_ckpt
    model_str = args.model
    use_adam_optimizer = args.use_adam_optimizer

    # Load images
    if not image_dir and not image_file:
        print('Provide one of the following options to load images \'-image_file\' or \'-image_dir\'')
        return None
    elif image_dir: 
        if not os.path.exists(image_dir):
            print('Image root directory \'%s\' not found' %image_dir)
            return None
        else:
            data = data_provider.load_images_by_subfolder(image_dir, VALIDATION_RATIO, SKIP_FOLDER, use_subfolder=USE_SUBFOLDER)
    else:
        if not os.path.exists(image_file):
            print('Image file \'%s\' not found' %image_file)
            return None
        else:
            data = data_provider.load_by_file(image_file, VALIDATION_RATIO)

    # Set a CNN model definition
    if model_str == 'vgg':
        model_def = VGG
    elif model_str == 'vgg_slim':
        model_def = VGGslim
    elif model_str == 'inc_v3':
        model_def = InceptionV3
    elif model_str == 'alex': # default
        model_def = AlexNet

    # Make sure the checkpoint dir exists
    ckpt_dir = os.path.join(CHECKPOINT_DIR, model_str)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # Start retraining/finetuning
    finetune(model_def, data, ckpt_dir, write_checkpoint_on_each_epoch, init_from_ckpt, use_adam_optimizer)

if __name__ == '__main__':
    main()