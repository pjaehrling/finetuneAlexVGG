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
from helper.feature.creator import FeatureCreator

# Input params
USE_SUBFOLDER = True
SKIP_FOLDER = ['yiwen']
BATCH_SIZE = 32

def create(image_paths, model_def, layer, feat_dir):
    """
    Args:
        image_paths:
        ckpt:
        model_def:
    """
    if not os.path.exists(feat_dir):
        os.makedirs(feat_dir)

    creator = FeatureCreator(
        model_def,
        feat_dir,
        image_paths['training_paths'],
        image_paths['training_labels'],
        image_paths['labels']
    )
    creator.run(
        layer, 
        BATCH_SIZE,
        use_train_prep=False, 
        memory_usage=1.
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
        '-model',
        type=str,
        choices=['vgg', 'vgg_slim', 'inc_v3''alex'],
        default='alex',
        help='Which model to use'
    )
    parser.add_argument(
        '-layer',
        type=str,
        default='fc6',
        help='Which layer to use as feature output'
    )
    parser.add_argument(
        '-feature_dir',
        type=str,
        default='features',
        help='Where to save the feature files'
    )
    
    args = parser.parse_args()
    image_dir = args.image_dir
    image_file = args.image_file
    model_str = args.model
    layer = args.layer
    feature_dir = args.feature_dir

    # Load images
    if not image_dir and not image_file:
        print('Provide one of the following options to load images \'-image_file\' or \'-image_dir\'')
        return None
    elif image_dir: 
        if not os.path.exists(image_dir):
            print('Image root directory \'%s\' not found' %image_dir)
            return None
        else:
            image_paths = data_provider.load_images_by_subfolder(image_dir, 0, SKIP_FOLDER, use_subfolder=USE_SUBFOLDER)
    else:
        if not os.path.exists(image_file):
            print('Image file \'%s\' not found' %image_file)
            return None
        else:
            image_paths = data_provider.load_by_file(image_file, 0)

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
    create(image_paths, model_def, layer, feature_dir)

if __name__ == '__main__':
    main()