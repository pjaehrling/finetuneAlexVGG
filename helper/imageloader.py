#
# Author: Philipp Jaehrling
#

import os
import re
import glob

import tensorflow as tf

from tensorflow.python.ops import math_ops
from tensorflow.python.framework.ops import convert_to_tensor

FILE_EXT = ['jpg'] # ['jpg', 'jpeg', 'JPG', 'JPEG']
MEAN = tf.constant([124, 117, 104], dtype=tf.float32) # IMAGENET

def get_images_in_folder(path, skip_folder, use_subfolder):
    """
    """
    images = []
    for ext in FILE_EXT:
        pattern = os.path.join(path, '*.' + ext)
        images += glob.glob(pattern)

    # check subfolder
    if use_subfolder:
        for subfolder in os.listdir(path):
            subfolder_path = os.path.join(path, subfolder)
            if os.path.isdir(subfolder_path) and not (subfolder in skip_folder):
                images += get_images_in_folder(subfolder_path, skip_folder, use_subfolder)
    return images

def load_image_paths_by_subfolder(root_dir, validation_ratio, skip_folder=[], load_as_tensor=True, use_subfolder=False):
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
        
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path) or folder in skip_folder:
            continue # skip files and skipped folders
        
        print('Looking for images in %s' %folder)

        image_paths = get_images_in_folder(folder_path, skip_folder, use_subfolder)
        if not image_paths:
            print('=> No image files found')
            continue # skip empty directories

        print('=> Found %i images' %len(image_paths))

        # split the list into traning and validation
        label = re.sub(r'[^a-z0-9]+', ' ', folder.lower())
        image_paths_sub = image_paths[::validation_ratio]
        del image_paths[::validation_ratio]

        # print infos
        print('  => Training: %i' %len(image_paths))
        print('  => Validation %i' %len(image_paths_sub))
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
        'training_paths': convert_to_tensor(training_paths, dtype=tf.string) if load_as_tensor else training_paths,
        'training_labels': convert_to_tensor(training_labels, dtype=tf.int32) if load_as_tensor else training_labels,
        'validation_image_count': len(validation_paths),
        'validation_paths': convert_to_tensor(validation_paths, dtype=tf.string) if load_as_tensor else validation_paths,
        'validation_labels': convert_to_tensor(validation_labels, dtype=tf.int32) if load_as_tensor else validation_labels
    }


def load_image_paths_by_file(image_file, validation_ratio, load_as_tensor=True):
    """
    Create a list of labeled images, seperated in training and validation sets.
    Reads the given 'image_file' line by line, where every line has the format *label* *image_path*

    Args:
        image_file: File with a list of labels and images
        validation_ratio: How much of the imagaes should go into the validation set

    Returns:
        A dictionary containing an entry for each subfolder/class, 
        with image-paths split into training and validation sets.
    """
    # Groupe the paths by label
    labeled_paths = {}
    f = open(image_file)
    for line in f: 
        line = line.strip().split(' ')
        image_label = line[0] 
        image_path = line[1]
        if image_label in labeled_paths:
            labeled_paths[image_label].append(image_path)
        else:
            labeled_paths[image_label] = [image_path]

    # Seperate them into training and validation sets 
    labels = []
    training_paths = []
    training_labels = []
    validation_paths = []
    validation_labels = []
    for label, image_paths in labeled_paths.items():
        print('Images with label \'%s\'' %label)
        print('=> Found %i images' %len(image_paths))

        # split the list into traning and validation
        image_paths_sub = image_paths[::validation_ratio]
        del image_paths[::validation_ratio]

        # print infos
        print('  => Training: %i' %len(image_paths))
        print('  => Validation %i' %len(image_paths_sub))

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
        'training_paths': convert_to_tensor(training_paths, dtype=tf.string) if load_as_tensor else training_paths,
        'training_labels': convert_to_tensor(training_labels, dtype=tf.int32) if load_as_tensor else training_labels,
        'validation_image_count': len(validation_paths),
        'validation_paths': convert_to_tensor(validation_paths, dtype=tf.string) if load_as_tensor else validation_paths,
        'validation_labels': convert_to_tensor(validation_labels, dtype=tf.int32) if load_as_tensor else validation_labels
    }