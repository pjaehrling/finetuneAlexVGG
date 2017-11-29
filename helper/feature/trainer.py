#
# Author: Philipp Jaehrling philipp.jaehrling@gmail.com)
# Influenced by: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/image_retraining
#

import math

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.data import Dataset

import helper.ops as ops
import helper.utils as utils
# from helper.layer import fc

class FeatureTrainer(object):
    """
    Train a given model (final_op) on a set of features 
    """

    def __init__(self, data, write_checkpoints = False):
        self.data = data # feature paths and lables
        self.num_classes = len(data['labels'])
        self.write_checkpoints = write_checkpoints

    @staticmethod
    def print_infos(train_vars, learning_rate, batch_size, keep_prob, use_adam):
        """Print infos about the current run

        Args: 
            train_vars:
            learning_rate:
            batch_size:
            keep_prob:
        """
        print("=> Will train:")
        for var in train_vars:
            print("  => {}".format(var))
        print("")
        print("=> Learningrate: %.4f" %learning_rate)
        print("=> Batchsize: %i" %batch_size)
        print("=> Dropout: %.4f" %(1.0 - keep_prob))
        print("=> Using Adam Optimizer: %r" %use_adam)
        print("")

    def parse_data(self, path, label):
        """
        Args:
            path:
            label:
            is_training:

        Returns:
            feature: feature as tensor
            label: converted label number into one-hot-encoding (binary)
        """
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        # load the image
        feat_file = tf.read_file(path)
        feat_vals = tf.string_split([feat_file], '\n')
        feat_floats = tf.string_to_number(feat_vals.values, out_type=tf.float32)
        return feat_floats, one_hot, path
    
    def create_dataset(self, is_training=True):
        """
        Args:
            is_training: Define what kind of Dataset should be returned (traning or validation)

        Returns: A Tensorflow Dataset with feaures and their labels. Either for training or validation.
        """
        paths = self.data['training_paths'] if is_training else self.data['validation_paths']
        labels = self.data['training_labels'] if is_training else self.data['validation_labels']
        dataset = Dataset.from_tensor_slices((
            tf.convert_to_tensor(paths, dtype=tf.string),
            tf.convert_to_tensor(labels, dtype=tf.int32)
        ))
        # setup the op to load the features
        dataset = dataset.map(self.parse_data)
        return dataset

    def create_model(self, layer_inputs, keep_prob, use_regularizer=False):
        """
        build model ...
        """
        ph_data = tf.placeholder(tf.float32, [None, layer_inputs[0]])
        ph_labels = tf.placeholder(tf.float32, [None, self.num_classes])
        
        layer = [ph_data]
        added = 0

        weights_regularizer = slim.l2_regularizer(0.001) if use_regularizer else None

        while added < len(layer_inputs):
            added += 1
            is_last = True if (added == len(layer_inputs)) else False
            
            # last layer output is defined by subfolder count (or classes in list of files)
            num_outputs = self.num_classes if is_last else layer_inputs[added]
            # no activation on last FC
            activation  = None if is_last else tf.nn.relu 

            # Add a new layer
            layer.append(
                tf.contrib.layers.fully_connected(         
                    inputs=layer[len(layer) - 1],
                    num_outputs=num_outputs,
                    activation_fn=activation,
                    # normalizer_fn=None,
                    # normalizer_params=None,
                    # weights_initializer=initializers.xavier_initializer(),
                    weights_regularizer=weights_regularizer,
                    # biases_initializer=tf.zeros_initializer(),
                    # biases_regularizer=None,
                    # reuse=None,
                    # variables_collections=None,
                    # outputs_collections=None,
                    trainable=True,
                    scope='fc%i' % added
                )
            )

            if not is_last: # no dropout on last FC
                tf.nn.dropout(layer[len(layer) - 1], keep_prob)

        # fc7 = tf.layers.dense(ph, 1024, name='fc7')
        # tf.layers.dense() does the same basically, but has a linear activation by default, both just higher api level functions
        # see: 
        # https://www.tensorflow.org/api_docs/python/tf/layers/dense
        # https://stackoverflow.com/questions/44912297/are-tf-layers-dense-and-tf-contrib-layers-fully-connected-interchangeable

        return ph_data, ph_labels, layer[added]

    ############################################################################
    def run(self, layer_inputs, epochs, learning_rate = 0.01, batch_size = 128, keep_prob = 1.0, memory_usage = 1.0, 
            device = '/gpu:0', save_ckpt_dir = '', init_ckpt_file = '', use_adam_optimizer=False, shuffle=True, use_regularizer=False):
        """
        Run training 

        Args:
            epochs:
            learning_rate:
            batch_size:
            keep_prob:
            memory_usage:
            device:
            show_misclassified:
            validate_on_each_epoch:
            save_ckpt_dir:
            init_ckpt_file:
        """
        # create datasets
        data_train = self.create_dataset(is_training=True)
        data_val = self.create_dataset(is_training=False)

        # Get ops to init the dataset iterators and get a next batch
        init_train_iterator_op, init_val_iterator_op, get_next_batch_op = ops.get_dataset_ops(
            data_train,
            data_val,
            batch_size,
            train_size=self.data['training_count'],
            val_size=self.data['validation_count'],
            shuffle=shuffle
        )

        # Initialize model and create input placeholders
        with tf.device(device):
            ph_keep_prob = tf.placeholder(tf.float32)
            ph_data, ph_labels, final_op = self.create_model(layer_inputs, keep_prob, use_regularizer)
        
        # Get a list with all trainable variables and print infos for the current run
        train_vars = tf.trainable_variables()
        self.print_infos(train_vars, learning_rate, batch_size, keep_prob, use_adam_optimizer)

        # Add/Get the different operations to optimize (loss, train and validate)
        with tf.device(device):
            loss_op = ops.get_loss_op(final_op, ph_labels)
            train_op = ops.get_train_op(loss_op, learning_rate, train_vars, use_adam_optimizer)
            accuracy_op, correct_prediction_op, predicted_index_op, true_index_op = ops.get_validation_ops(final_op, ph_labels)

        # Get the number of training/validation steps per epoch to get through all images
        batches_per_epoch_train = int(math.ceil(self.data['training_count'] / (batch_size + 0.0)))
        batches_per_epoch_val   = int(math.ceil(self.data['validation_count'] / (batch_size + 0.0)))

        # Initialize a saver, create a session config and start a session
        saver = tf.train.Saver()
        gpu_options = tf.GPUOptions()
        gpu_options.per_process_gpu_memory_fraction = memory_usage
        gpu_options.visible_device_list="1"
        config = tf.ConfigProto(gpu_options=gpu_options)
        server = tf.train.Server.create_local_server(config=config)
        
        with tf.Session(target=server.target) as sess:
            # Init all variables 
            sess.run(tf.global_variables_initializer())
            
            # Load the pretrained variables or a saved checkpoint
            if init_ckpt_file:
                saver.restore(sess, init_ckpt_file)

            utils.print_output_header(self.data['training_count'], self.data['validation_count'])
            for epoch in range(epochs):
                is_last_epoch = True if (epoch+1) == epochs else False
                
                train_loss, train_acc = utils.run_training(
                    sess,
                    train_op,
                    loss_op,
                    accuracy_op,
                    init_train_iterator_op,
                    get_next_batch_op,
                    ph_data,
                    ph_labels,
                    ph_keep_prob,
                    keep_prob,
                    batches_per_epoch_train
                )
                
                return_misclassified = is_last_epoch
                test_loss, test_acc, misclassified = utils.run_validation(
                    sess,
                    loss_op,
                    accuracy_op,
                    correct_prediction_op,
                    predicted_index_op,
                    true_index_op,
                    final_op,
                    init_val_iterator_op,
                    get_next_batch_op,
                    ph_data,
                    ph_labels,
                    ph_keep_prob,
                    batches_per_epoch_val,
                    return_misclassified
                )

                utils.print_output_epoch(epoch + 1, train_loss, train_acc, test_loss, test_acc)

                # show missclassified list on last epoch
                if is_last_epoch:
                    utils.print_misclassified(sess, misclassified, self.data['labels'])

                # save session in a checkpoint file
                if self.write_checkpoints or is_last_epoch:
                    utils.save_session_to_checkpoint_file(sess, saver, epoch, save_ckpt_dir)
