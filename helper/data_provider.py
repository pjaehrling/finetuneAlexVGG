#
# Author: Philipp Jaehrling
#

import os
import re
import glob

def get_files_in_folder(path, file_extensions, skip_folder, use_subfolder):
    """
    """
    files = []
    for ext in file_extensions:
        pattern = os.path.join(path, '*.' + ext)
        files += glob.glob(pattern)

    # check subfolder
    if use_subfolder:
        for subfolder in os.listdir(path):
            subfolder_path = os.path.join(path, subfolder)
            if os.path.isdir(subfolder_path) and not (subfolder in skip_folder):
                files += get_files_in_folder(subfolder_path, file_extensions, skip_folder, use_subfolder)
    return files

def load_images_by_subfolder(root_dir, validation_ratio, skip_folder=[], use_subfolder=False):
    file_extensions =  ['jpg'] # ['jpg', 'jpeg']
    return load_by_subfolder(root_dir, file_extensions, validation_ratio, skip_folder, use_subfolder)

def load_features_by_subfolder(root_dir, validation_ratio, skip_folder=[], use_subfolder=False):
    file_extensions =  ['txt']
    return load_by_subfolder(root_dir, file_extensions, validation_ratio, skip_folder, use_subfolder)

def load_by_subfolder(root_dir, file_extensions, validation_ratio, skip_folder=[], use_subfolder=False):
    """
    Create a list of labeled data, seperated in training and validation sets.
    Will create a new label/class for every sub-directory in the 'root_dir'.

    Args:
        root_dir: String path to a folder containing subfolders with data.
        validation_ratio: How much of the data should go into the validation set

    Returns:
        A dictionary containing an entry for each subfolder/class, 
        with paths split into training and validation sets.
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
        
        print('Looking for data in %s' %folder)

        paths = get_files_in_folder(folder_path, file_extensions, skip_folder, use_subfolder)
        if not paths:
            print('=> No files found')
            continue # skip empty directories

        print('=> Found %i entries' %len(paths))

        # split the list into traning and validation
        label = re.sub(r'[^a-z0-9]+', ' ', folder.lower())
        if (validation_ratio > 0):
            paths_sub = paths[::validation_ratio]
            del paths[::validation_ratio]
        else:
            paths_sub = []

        # print infos
        print('  => Training: %i' %len(paths))
        print('  => Validation %i' %len(paths_sub))
        print('  => Labeling them with: {} ({})'.format(label, len(labels)))

        # add entries to the result
        labels.append(label)
        label_index = len(labels) - 1
        training_paths += paths
        training_labels += [label_index] * len(paths)
        validation_paths += paths_sub
        validation_labels += [label_index] * len(paths_sub)

    return {
        'labels': labels,
        'training_count': len(training_paths),
        'training_paths': training_paths,
        'training_labels': training_labels,
        'validation_count': len(validation_paths),
        'validation_paths': validation_paths,
        'validation_labels': validation_labels
    }

def load_by_file(file, validation_ratio):
    """
    Create a list of labeled data, seperated in training and validation sets.
    Reads the given 'file' line by line, where every line has the format *label* *data_path*

    Args:
        file: File with a list of labels and data path
        validation_ratio: How much of the data should go into the validation set

    Returns:
        A dictionary containing an entry for each subfolder/class, 
        with paths split into training and validation sets.
    """
    # Groupe the paths by label
    labeled_paths = {}
    f = open(file)
    for line in f: 
        line = line.strip().split(' ')
        label = line[0] 
        path = line[1]
        if label in labeled_paths:
            labeled_paths[label].append(path)
        else:
            labeled_paths[label] = [path]

    # Seperate them into training and validation sets 
    labels = []
    training_paths = []
    training_labels = []
    validation_paths = []
    validation_labels = []
    for label, paths in labeled_paths.items():
        print('Data with label \'%s\'' %label)
        print('=> Found %i entries' %len(paths))

        # split the list into traning and validation
        if (validation_ratio > 0):
            paths_sub = paths[::validation_ratio]
            del paths[::validation_ratio]
        else:
            paths_sub = []

        # print infos
        print('  => Training: %i' %len(paths))
        print('  => Validation %i' %len(paths_sub))

        # add entries to the result
        labels.append(label)
        label_index = len(labels) - 1
        training_paths += paths
        training_labels += [label_index] * len(paths)
        validation_paths += paths_sub
        validation_labels += [label_index] * len(paths_sub)

    return {
        'labels': labels,
        'training_count': len(training_paths),
        'training_paths': training_paths,
        'training_labels': training_labels,
        'validation_count': len(validation_paths),
        'validation_paths': validation_paths,
        'validation_labels': validation_labels
    }