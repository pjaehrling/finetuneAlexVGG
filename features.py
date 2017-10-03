#
# Author: Philipp Jaehrling
# Influenced by: 
# - https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html
# - https://kratzert.github.io/2017/06/15/example-of-tensorflows-new-input-pipeline.html
#
import os
import argparse

from models.alexnet import AlexNet
from models.vgg import VGG
from models.vgg_slim import VGGslim
from models.inception_v3 import InceptionV3
from helper.imageloader import load_image_paths_by_subfolder, load_image_paths_by_file
from helper.featurecreator import FeatureCreator

# Input params
USE_SUBFOLDER = True
SKIP_FOLDER = ['yiwen']

# Params
FEAT_LAYER = 'fc6'

def create(image_paths, model_def, feat_dir):
    """
    Args:
        image_paths:
        ckpt:
        model_def:
    """
    if not os.path.exists(feat_dir):
        os.makedirs(feat_dir)

    creator = FeatureCreator(model_def, image_paths['training_paths'], feat_dir)
    creator.run(
        FEAT_LAYER, 
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
        default='alex',
        help='Model to be validated'
    )
    
    args = parser.parse_args()
    image_dir = args.image_dir
    image_file = args.image_file
    model_str = args.model

    # Load images
    if not image_dir and not image_file:
        print('Provide one of the following options to load images \'-image_file\' or \'-image_path\'' %image_dir)
        return None
    elif image_dir: 
        if not os.path.exists(image_dir):
            print('Image root directory \'%s\' not found' %image_dir)
            return None
        else:
            image_paths = load_image_paths_by_subfolder(image_dir, 0, SKIP_FOLDER, use_subfolder=USE_SUBFOLDER)
    else:
        if not os.path.exists(image_file):
            print('Image file \'%s\' not found' %image_file)
            return None
        else:
            image_paths = load_image_paths_by_file(image_file, 0)

    # Set a CNN model definition
    if model_str == 'vgg':
        model_def = VGG
    elif model_str == 'vgg_slim':
        model_def = VGGslim
    elif model_str == 'inc_v3':
        model_def = InceptionV3
    elif model_str == 'alex': # default
        model_def = AlexNet

    # Feature dir
    feat_dir = os.path.join('../features', model_str)

    # Start retraining/finetuning
    create(image_paths, model_def, feat_dir)

if __name__ == '__main__':
    main()