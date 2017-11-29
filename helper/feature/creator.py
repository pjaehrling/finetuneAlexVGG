#
# Author: Philipp Jaehrling (philipp.jaehrling@gmail.com)
#
import os
import math
import hashlib

# Tensorflow imports
import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator

class FeatureCreator(object):
    """
    """

    def __init__(self, model_def, feature_dir, image_paths, image_labels, label_dict, image_prep=None, is_resnet=False):
        self.model_def = model_def
        self.is_resnet = is_resnet
        self.image_prep = image_prep if image_prep else model_def.image_prep # overwrite the default model image prep? 

        # Create a list with feature paths (image -> feat)
        self.image_paths = []
        self.feature_paths = []
        mappings = []
        existed = []

        for i, image_path in enumerate(image_paths):
            label = label_dict[image_labels[i]]
            feature_path = self.get_feature_path(feature_dir, image_path, label)
            
            if not os.path.exists(feature_path): # just add what we have to create
                self.feature_paths.append(feature_path)
                self.image_paths.append(image_path)
            else:
                existed.append(image_path)

            mapping = "{} -> {}".format(image_path, feature_path)
            mappings.append(mapping)

        print("=> Skipped {} entries because they already existed".format(len(existed)))
        self.count = len(self.image_paths)
        self.write_mapping_file(feature_dir, mappings)

    @staticmethod
    def get_feature_path(feature_dir, image_path, label):
        """
        Args:
            image_path
        """
        feature_dir = os.path.join(feature_dir, label)
        if not os.path.exists(feature_dir):
            os.makedirs(feature_dir)

        hashed_path = hashlib.md5(image_path.encode()).hexdigest()
        bn_path = os.path.join(feature_dir, hashed_path + '.txt')
        return bn_path

    def write_mapping_file(self, path, mappings):
        """
        """
        mapping_path = os.path.join(path, 'mapping.txt')
        mapping_file = open(mapping_path, "w")
        mapping_file.write('\n'.join(mappings))
        mapping_file.close()

    def parse_data(self, img_path, feat_path, use_train_prep):
        """
        Args:
            path:
            feat_path:
            use_training_preprocessing:

        Returns:
            image: image loaded and preprocesed
            label: converted label number into one-hot-encoding (binary)
        """
        # load the image
        img_file      = tf.read_file(img_path)
        img_decoded   = tf.image.decode_jpeg(img_file, channels=3)
        img_processed = self.image_prep.preprocess_image(
            image=img_decoded,
            output_height=self.model_def.image_size,
            output_width=self.model_def.image_size,
            is_training=use_train_prep
        )
        return img_processed, feat_path

    def parse_train_data(self, path, label):
        return self.parse_data(path, label, True)

    def parse_validation_data(self, path, label):
        return self.parse_data(path, label, False)

    def create_dataset(self, use_train_prep=False):
        """
        Args:
            use_training_preprocessing: use training mode for image preprocessing

        Returns: A Tensorflow Dataset with images and feature paths.
        """        
        dataset = Dataset.from_tensor_slices((
            tf.convert_to_tensor(self.image_paths, dtype=tf.string),
            tf.convert_to_tensor(self.feature_paths, dtype=tf.string)
        ))

        # load and preprocess the images
        if (use_train_prep):
            dataset = dataset.map(self.parse_train_data)
        else:
            dataset = dataset.map(self.parse_validation_data)
        return dataset

    def create_feat_files(self, sess, feat_tensor, ph_image, iterator):
        """
        Args:
            sess:
            feat_tensor:
            ph_image:
            iterator:
        """
        images, feat_paths = sess.run(iterator.get_next()) # get next batch
        features = sess.run(feat_tensor, feed_dict={ph_image: images} )

        for i, feat in enumerate(features):
            feat_flat = feat.flatten()
            feat_string = '\n'.join("{:.16f}".format(x) for x in feat_flat)
            with open(feat_paths[i], 'w') as feat_file:
                feat_file.write(feat_string)

    @staticmethod
    def get_feature_tensor(endpoints, feat_layer):
        for key, tensor in endpoints.items():
            if key.endswith(feat_layer):
                return key, tensor
        return '', None

    def run(self, feat_layer, batch_size, use_train_prep=False, memory_usage=1.):
        """
        Args:
            feat_layer:
            use_train_prep:
            memory_usage:
        """
        if self.count is 0: return

        # create datasets
        data = self.create_dataset(use_train_prep)
        data = data.batch(batch_size)
        iterator = data.make_one_shot_iterator()
        batches = int(math.ceil(self.count / (batch_size + 0.0)))

        # Initialize model and create input placeholders
        ph_image = tf.placeholder(tf.float32, [None, self.model_def.image_size, self.model_def.image_size, 3])

        if self.is_resnet:
            # Init ResNet and get the final output (which is not part of the endpoints dict)
            # -> when "num_classes=None" we get the last Residential block aka convolution (after global average pooling) output
            model = self.model_def(ph_image, num_classes=None) 
            feat_tensor = model.get_final_op()
            name = 'average pooling'
        else:
            # Init the model and get all the endpoints
            model = self.model_def(ph_image)
        
            endpoints = model.get_endpoints()
            name, feat_tensor = self.get_feature_tensor(endpoints, feat_layer)
        
            if feat_tensor is None:
                print("=> Couldn't find matching layer. Available:")
                for key, tensor in endpoints.items():
                    print("  => " + key)
                    print(tensor)
                return

        # Go on the same way for ResNet and all others
        print("=> Loading data from layer: %s" %name)
        print(feat_tensor)

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = memory_usage
        with tf.Session(config=config) as sess:
            print("=> Session started ...")
            # Init all variables 
            sess.run(tf.global_variables_initializer())
            model.load_initial_weights(sess)

            print("=> Start creating features ...")
            for batch_step in range(batches):
                self.create_feat_files(sess, feat_tensor, ph_image, iterator)
                print("  => %i done" %((batch_step+1)*batch_size))