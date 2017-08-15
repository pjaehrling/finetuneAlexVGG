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

from alexnet import AlexNet
from caffe_classes import class_names
from tensorflow.contrib.data import Dataset, Iterator
from tensorflow.python.ops import math_ops

#
# load images
#
def load_images():
    print "loading images ..."
    img_dir = os.path.join('.', 'images')
    files = fnmatch.filter(os.listdir(img_dir), '*.jpeg')

    images = []
    for f in files:
        print "> " + f
        img_file    = tf.read_file(os.path.join(img_dir, f))
        img_decoded = tf.image.decode_jpeg(img_file, channels=3)

        # A) cropped
        # img_resized = tf.image.resize_image_with_crop_or_pad(img_decoded, 227, 227) 
        # B) resized
        input_size = tf.constant([227, 227], dtype=tf.int32)
        img_resized = tf.image.resize_images(img_decoded, input_size)
        
        # A) calulates the mean for every image separately
        # img_standardized = tf.image.per_image_standardization(img_resized) 
        
        # B) Subtract the imagenet mean (mean over all imagenet images)
        imgnet_mean = tf.constant([124, 127, 104], dtype=tf.float32)
        imgnet_mean = tf.reshape(imgnet_mean, [1, 1, 3])
        img_cast = math_ops.cast(img_resized, dtype=tf.float32)
        img_standardized = math_ops.subtract(img_cast, imgnet_mean)

        # in this alexnet implementation the images are feed to the net in BGR format, NOT RGB
        channels = tf.unstack(img_standardized, axis=-1)
        img_standardized  = tf.stack([channels[2], channels[1], channels[0]], axis=-1)

        images.append(img_standardized)

    return Dataset.from_tensors(images)

#
# validate my alexnet implementation
#
def validate():
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