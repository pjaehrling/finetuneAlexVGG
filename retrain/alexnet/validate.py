#
# Author: Philipp Jaehrling
# Influenced by: 
# - https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html
# - https://kratzert.github.io/2017/06/15/example-of-tensorflows-new-input-pipeline.html
#

import os
import fnmatch
import numpy as np
import tensorflow as tf
import imageloader as imgl

from alexnet import AlexNet
from caffe_classes import class_names
from tensorflow.contrib.data import Dataset, Iterator
from tensorflow.python.ops import math_ops


def load_images():
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
        img = imgl.load_img_as_tensor(
            os.path.join(img_dir, f),
            input_width=227,
            input_height=227,
            crop=False,
            use_mean=True,
            bgr=True
        )
        images.append(img)

    return Dataset.from_tensors(images)


def validate():
    """
    Validate my alexnet implementation
    """

    # load the images
    images = load_images()

    # create TensorFlow Iterator object
    iterator = Iterator.from_structure(images.output_types, images.output_shapes)
    next_element = iterator.get_next()
    iterator_init_op = iterator.make_initializer(images)
    
    dropout = tf.constant(1, dtype=tf.float32)

    # create model with default config
    model = AlexNet(next_element, dropout, num_classes=1000, skip_layer=[])
    # create op to calculate softmax (fc8 is last layer in alexnet impl.)
    softmax = tf.nn.softmax(model.fc8)

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
    validate()

if __name__ == '__main__':
    main()