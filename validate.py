#
# Author: Philipp Jaehrling
# Influenced by: 
# - https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html
# - https://kratzert.github.io/2017/06/15/example-of-tensorflows-new-input-pipeline.html
#

import os
import fnmatch
import argparse
import numpy as np
import tensorflow as tf

from models.alexnet import AlexNet
from models.vgg import VGG
from helper.imagenet_classes import class_names
from helper.imageloader import load_img_as_tensor

from tensorflow.contrib.data import Dataset, Iterator
from tensorflow.python.ops import math_ops

def load_images(model_def):
    """
    Load images (BGR) from ./images folder
    - Resizes/Croppes the images to 227 x 227 px
    - Subtracts the ImageNet mean

    Returns:
        Tensorflow Dataset with the images (BGR) loaded as tensors
    """
    print "loading images ..."
    img_dir = os.path.join('.', 'images')
    files = fnmatch.filter(os.listdir(img_dir), '*.jpeg')

    images = []
    for f in files:
        print "> " + f
        img = load_img_as_tensor(
            os.path.join(img_dir, f),
            input_width=model_def.input_width,
            input_height=model_def.input_height,
            crop=False,
            sub_in_mean=model_def.subtract_imagenet_mean,
            bgr=model_def.use_bgr
        )
        images.append(img)

    return Dataset.from_tensors(images)


def validate(model_def):
    """
    Validate my alexnet implementation

    Args:
        model_def: the model class/definition
    """

    # load the images
    images = load_images(model_def)

    # create TensorFlow Iterator object
    iterator = Iterator.from_structure(images.output_types, images.output_shapes)
    next_element = iterator.get_next()
    iterator_init_op = iterator.make_initializer(images)
    
    dropout = tf.constant(1, dtype=tf.float32)

    # create model with default config
    model = model_def(next_element, dropout, num_classes=1000, retrain_layer=[])
    # create op to calculate softmax
    softmax = tf.nn.softmax(model.final)

    print 'start validation ...'
    with tf.Session() as sess:
    
        # Initialize all variables and the iterator
        sess.run(tf.global_variables_initializer())
        sess.run(iterator_init_op)
    
        # Load the pretrained weights into the model
        model.load_initial_weights(sess)

        # run the graph
        probs = sess.run(softmax)
        
        # print the results
        for prob in probs:
            best_index = np.argmax(prob)
            print "> " + class_names[best_index] + " -> %.4f" %prob[best_index]


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-model',
        type=str,
        default='alex',
        help='Model to be validated'
    )
    args = parser.parse_args()
    model_str = args.model

    if model_str == 'vgg':
        model_def = VGG
    elif model_str == 'alex': # default
        model_def = AlexNet

    validate(model_def)

if __name__ == '__main__':
    main()