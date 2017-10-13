#
# Author: Philipp Jaehrling philipp.jaehrling@gmail.com)
# Influenced by: https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html
#

import math
from datetime import datetime

import tensorflow as tf
from tensorflow.contrib.data import Dataset

import helper.ops as ops
import helper.utils as utils

class Retrainer(object):
    """
    Retrain (Finetune) a given model on a new set of categories 
    """

    def __init__(self, model_def, data, write_summary = False, summary_step = 1, summary_path = '', write_checkpoints = False):
        self.model_def = model_def
        self.data = data
        self.num_classes = len(data['labels'])

        self.write_summary = write_summary
        self.summary_step = summary_step
        self.summary_path = summary_path
        self.write_checkpoints = write_checkpoints

    @staticmethod
    def print_infos(train_vars, restore_vars, learning_rate, batch_size, keep_prob):
        """Print infos about the current run

        Args:
            restore_vars: 
            train_vars:
            learning_rate:
            batch_size:
            keep_prob:
        """
        print("=> Will Restore:")
        for var in restore_vars:
            print("  => {}".format(var))
        print("=> Will train:")
        for var in train_vars:
            print("  => {}".format(var))
        print("=> Learningrate: %.4f" %learning_rate)
        print("=> Batchsize: %i" %batch_size)
        print("=> Dropout: %.4f" %(1.0 - keep_prob))
        print("##################################")

    def parse_data(self, path, label, is_training):
        """
        Args:
            path:
            label:
            is_training:

        Returns:
            image: image loaded and preprocesed
            label: converted label number into one-hot-encoding (binary)
        """
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        # load the image
        img_file      = tf.read_file(path)
        img_decoded   = tf.image.decode_jpeg(img_file, channels=3)
        img_processed = self.model_def.image_prep.preprocess_image(
            image=img_decoded,
            output_height=self.model_def.image_size,
            output_width=self.model_def.image_size,
            is_training=is_training
        )
        return img_processed, one_hot

    def parse_train_data(self, path, label):
        return self.parse_data(path, label, True)

    def parse_validation_data(self, path, label):
        return self.parse_data(path, label, False)
    
    def create_dataset(self, is_training=True):
        """
        Args:
            is_training: Define what kind of Dataset should be returned (traning or validation)

        Returns: A Tensorflow Dataset with images and their labels. Either for training or validation.
        """
        paths = self.data['training_paths'] if is_training else self.data['validation_paths']
        labels = self.data['training_labels'] if is_training else self.data['validation_labels']
        dataset = Dataset.from_tensor_slices((
            tf.convert_to_tensor(paths, dtype=tf.string),
            tf.convert_to_tensor(labels, dtype=tf.int32)
        ))
        # load and preprocess the images
        if is_training:
            dataset = dataset.map(self.parse_train_data)
        else:
            dataset = dataset.map(self.parse_validation_data)

        return dataset

    ############################################################################
    def run(self, finetune_layers, epochs, learning_rate = 0.01, batch_size = 128, keep_prob = 1.0, memory_usage = 1.0, 
            device = '/gpu:0', show_misclassified = False, validate_on_each_epoch = False, save_ckpt_dir = '', init_ckpt_file = ''):
        """
        Run a training on part of the model (retrain/finetune)

        Args:
            finetune_layers:
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
        init_train_iterator_op, init_val_iterator_op, get_next_batch_op = ops.get_dataset_ops(data_train, data_val, batch_size)

        # Initialize model and create input placeholders
        with tf.device(device):
            ph_images = tf.placeholder(tf.float32, [batch_size, self.model_def.image_size, self.model_def.image_size, 3])
            ph_labels = tf.placeholder(tf.float32, [batch_size, self.num_classes])
            ph_keep_prob = tf.placeholder(tf.float32)

            model = self.model_def(ph_images, keep_prob=ph_keep_prob, num_classes=self.num_classes, retrain_layer=finetune_layers)
            final_op = model.get_final_op()
        
        # Get a list with all trainable variables and print infos for the current run
        retrain_vars = model.get_retrain_vars()
        restore_vars = model.get_restore_vars()
        self.print_infos(retrain_vars, restore_vars, learning_rate, batch_size, keep_prob)

        # Add/Get the different operations to optimize (loss, train and validate)
        with tf.device(device):
            loss_op = ops.get_loss_op(final_op, ph_labels)
            train_op = ops.get_train_op(loss_op, learning_rate, retrain_vars)
            accuracy_op, predicted_index_op = ops.get_validation_ops(final_op, ph_labels)

        # Create operat
        summary_op = None
        writer = None
        if self.write_summary:
            summary_op, writer = ops.get_summary_writer_op(retrain_vars, loss_op, accuracy_op, self.summary_path)

        # Get the number of training/validation steps per epoch to get through all images
        batches_per_epoch_train = int(math.floor(self.data['training_count'] / (batch_size + 0.0)))
        batches_per_epoch_val   = int(math.floor(self.data['validation_count'] / (batch_size + 0.0)))
        # TODO: For the leftover images, change shape of the placeholder and use them? (leftovers are not used atm)

        # Initialize a saver, create a session config and start a session
        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = memory_usage
        with tf.Session(config=config) as sess:
            # Init all variables 
            sess.run(tf.global_variables_initializer())
            
            # Add the model graph to TensorBoard and print infos
            if self.write_summary: 
                writer.add_graph(sess.graph)
            
            # Load the pretrained variables or a saved checkpoint
            if init_ckpt_file:
                saver.restore(sess, init_ckpt_file)
            else: 
                model.load_initial_weights(sess)

            for epoch in range(epochs):
                print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
                is_last_epoch = True if (epoch+1) == epochs else False
                
                utils.run_training(
                    sess,
                    train_op,
                    init_train_iterator_op,
                    get_next_batch_op,
                    ph_images,
                    ph_labels,
                    ph_keep_prob,
                    keep_prob,
                    batches_per_epoch_train,
                    epoch,
                    self.write_summary,
                    summary_op,
                    writer,
                    self.summary_step
                )

                if validate_on_each_epoch or is_last_epoch:
                    utils.run_validation(
                        sess,
                        accuracy_op,
                        predicted_index_op,
                        init_val_iterator_op,
                        get_next_batch_op,
                        ph_images,
                        ph_labels,
                        ph_keep_prob,
                        batches_per_epoch_val,
                        batch_size,
                        epoch,
                        is_last_epoch,
                        show_misclassified,
                        self.data
                    )

                # save session in a checkpoint file
                if self.write_checkpoints or is_last_epoch:
                    utils.save_session_to_checkpoint_file(sess, saver, epoch, save_ckpt_dir)
