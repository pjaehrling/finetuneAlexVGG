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
from models.vgg_slim import VGGslim
from models.inception_v3 import InceptionV3
from models.resnet_v2 import ResNetV2
from helper.imagenet_classes import class_names

from tensorflow.contrib.data import Dataset, Iterator

def prep_resnet_results(probs):
    """
    For ResNet the result is a rank-4 tensor of size [images, 1, 1, num_classes].
    """
    return [prob[0][0] for prob in probs]

def validate(model_def):
    """
    Validate my alexnet implementation

    Args:
        model_def: the model class/definition
    """

    img_dir = os.path.join('.', 'images')
    images = []

    print "loading images ..."
    files = fnmatch.filter(os.listdir(img_dir), '*.jpeg')
    for f in files:
        print "> " + f
        img_file      = tf.read_file(os.path.join(img_dir, f))
        img_decoded   = tf.image.decode_jpeg(img_file, channels=3)
        img_processed = model_def.image_prep.preprocess_image(
            image=img_decoded, 
            output_height=model_def.image_size,
            output_width=model_def.image_size,
            is_training=False
        )
        images.append(img_processed)

    # create TensorFlow Iterator object
    images = Dataset.from_tensors(images)
    iterator = Iterator.from_structure(images.output_types, images.output_shapes)
    next_element = iterator.get_next()
    iterator_init_op = iterator.make_initializer(images)

    # create the model and get scores (pipe to softmax)
    model = model_def(next_element)
    scores = model.get_final_op()
    softmax = tf.nn.softmax(scores)

    print 'start validation ...'
    with tf.Session() as sess:
    
        # Initialize all variables and the iterator
        sess.run(tf.global_variables_initializer())
        sess.run(iterator_init_op)
    
        # Load the pretrained weights into the model
        model.load_initial_weights(sess)

        # run the graph
        probs = sess.run(softmax)

        if model_def is ResNetV2:
            probs = prep_resnet_results(probs)

        # sometime we have an offset
        offset = len(class_names) - len(probs[0])
        
        # print the results
        for prob in probs:
            best_index = np.argmax(prob)
            print "> " + class_names[best_index+offset] + " -> %.4f" %prob[best_index]

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
    elif model_str == 'vgg_slim':
        model_def = VGGslim
    elif model_str == 'inc_v3':
        model_def = InceptionV3
    elif model_str == 'res_v2':
        model_def = ResNetV2
    elif model_str == 'alex': # default
        model_def = AlexNet

    validate(model_def)

if __name__ == '__main__':
    main()