#
# Author: Philipp Jaehrling (philipp.jaehrling@gmail.com)
#
import os
import hashlib

# Tensorflow imports
import tensorflow as tf
from tensorflow.contrib.data import Dataset

class FeatureCreator(object):
    """
    """

    def __init__(self, model_def, image_paths, feature_dir):
        self.model_def = model_def
        self.feature_dir = feature_dir
        self.image_paths = image_paths
        self.feature_paths = [self.get_feature_path(p) for p in image_paths]
        self.count = len(image_paths)

    def get_feature_path(self, image_path):
        """
        Args:
            image_path
        """
        hashed_path = hashlib.md5(image_path.encode()).hexdigest()
        bn_path = os.path.join(self.feature_dir, hashed_path + '.txt')
        return bn_path

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
        img_processed = self.model_def.image_prep.preprocess_image(
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

    def create_feat_file(self, sess, feat_tensor, ph_image, iterator):
        """
        Args:
            sess:
            feat_tensor:
            ph_image:
            iterator:
        """
        image, feat_path = sess.run(iterator.get_next())
        
        if not os.path.exists(feat_path):
            feat = sess.run(feat_tensor, feed_dict={ph_image: [image]} )

            feat_flat = feat.flatten()
            feat_string = ','.join(str(x) for x in feat_flat)
            with open(feat_path, 'w') as feat_file:
                feat_file.write(feat_string)

    def run(self, feat_layer, use_train_prep=False, memory_usage=1.):
        """
        Args:
            feat_layer:
            use_train_prep:
            memory_usage:
        """
        # create datasets
        data = self.create_dataset(use_train_prep)
        iterator = data.make_one_shot_iterator()

        # Initialize model and create input placeholders
        ph_image = tf.placeholder(tf.float32, [1, self.model_def.image_size, self.model_def.image_size, 3])

        # Init the model and get all the endpoints
        model = self.model_def(ph_image)
        endpoints = model.get_endpoints()

        # Get the tensor, which should give us the output for the feature
        feat_tensor_index = [i for i, item in enumerate(endpoints.keys()) if item.endswith(feat_layer)][0]
        feat_tensor = endpoints.values()[feat_tensor_index]

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = memory_usage
        with tf.Session(config=config) as sess:
            # Init all variables 
            sess.run(tf.global_variables_initializer())
            model.load_initial_weights(sess)

            print("=> Start creating features ...")
            for i in range(self.count):
                self.create_feat_file(sess, feat_tensor, ph_image, iterator)
                if ((i+1)%100 == 0): print("  => %i done" %i)