#
# Author: Philipp Jaehrling
#

import os
import re

import tensorflow as tf

from tensorflow.python.ops import math_ops
from tensorflow.python.platform import gfile
from tensorflow.python.framework.ops import convert_to_tensor

FILE_EXT = ['jpg', 'jpeg', 'JPG', 'JPEG']
MEAN = tf.constant([124, 117, 104], dtype=tf.float32) # IMAGENET
# MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)

def load_image_paths_by_subfolder(root_dir, validation_ratio):
    """
    Create a list of labeled images, seperated in training and validation sets.
    Will create a new label/class for every sub-directory in the 'root_dir'.

    Args:
        root_dir: String path to a folder containing subfolders of images.
        validation_ratio: How much of the imagaes should go into the validation set

    Returns:
        A dictionary containing an entry for each subfolder/class, 
        with image-paths split into training and validation sets.
    """
    labels = []
    training_paths = []
    training_labels = []
    validation_paths = []
    validation_labels = []
        
    for d in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, d)
        if not os.path.isdir(dir_path): 
            continue # skip this file

        # get directory name and log progress
        dir_name = os.path.basename(dir_path)
        print 'Looking for images in %s' %dir_name

        # get all image files in this directory
        patterns = (os.path.join(dir_path, '*.' + ext) for ext in FILE_EXT)
        image_paths = gfile.Glob(patterns) # Returns a list of paths that match the given pattern(s)
        if not image_paths:
            print '=> No image files found'
            continue # skip empty directories

        print '=> Found %i images' %len(image_paths)

        # split the list into traning and validation
        label = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        image_paths_sub = image_paths[::validation_ratio]
        del image_paths[::validation_ratio]

        # print infos
        print '  => Training: %i' %len(image_paths)
        print '  => Validation %i' %len(image_paths_sub)
        print('  => Labeling them with: {} ({})'.format(label, len(labels)))

        # add entries to the result
        labels.append(label)
        label_index = len(labels) - 1
        training_paths += image_paths
        training_labels += [label_index] * len(image_paths)
        validation_paths += image_paths_sub
        validation_labels += [label_index] * len(image_paths_sub)

    return {
        'labels': labels,
        'training_image_count': len(training_paths),
        'training_paths': convert_to_tensor(training_paths, dtype=tf.string),
        'training_labels': convert_to_tensor(training_labels, dtype=tf.int32),
        'validation_image_count': len(validation_paths),
        'validation_paths': convert_to_tensor(validation_paths, dtype=tf.string),
        'validation_labels': convert_to_tensor(validation_labels, dtype=tf.int32)
    }



def load_img_as_tensor(path, input_width, input_height, crop = False, sub_in_mean = False, bgr = False):
    """
    Args:
        path: String path to the image that should be loaded

    Returns:
        A tensor containing the image (resized/cropped and standardized) with it's label
    """
    img_file    = tf.read_file(path)
    img_decoded = tf.image.decode_jpeg(img_file, channels=3)

    # Resize / Crop
    if crop:
        img_resized = tf.image.resize_image_with_crop_or_pad(img_decoded, input_width, input_height) 
    else:
        input_size = tf.constant([input_width, input_height], dtype=tf.int32)
        img_resized = tf.image.resize_images(img_decoded, input_size)
        
    # Normalise
    if sub_in_mean:
        # Subtract the imagenet mean (mean over all imagenet images)
        imgnet_mean = tf.reshape(MEAN, [1, 1, 3])
        img_cast = math_ops.cast(img_resized, dtype=tf.float32)
        img_standardized = math_ops.subtract(img_cast, imgnet_mean)
    else:
        # calulates the mean for every image separately (if I got this right)
        img_standardized = tf.image.per_image_standardization(img_resized)

    if bgr:
        # e.g. in my alexnet implementation the images are feed to the net in BGR format, NOT RGB
        channels = tf.unstack(img_standardized, axis=-1)
        img_standardized  = tf.stack([channels[2], channels[1], channels[0]], axis=-1)

    return img_standardized