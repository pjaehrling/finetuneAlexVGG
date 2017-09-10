#
# Author: Philipp Jaehrling philipp.jaehrling@gmail.com)
# Influenced by: https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html
#

import os
import math
import tensorflow as tf

from models.alexnet import AlexNet
from models.vgg import VGG
from helper.imageloader import load_img_as_tensor

from datetime import datetime
from tensorflow.contrib.data import Dataset, Iterator

# Tensorflow/board params
WRITE_SUMMARY = False
WRITE_CHECKPOINTS = True
DISPLAY_STEP = 20 # How often to write the tf.summary
TENSORBOARD_PATH = "/Users/philipp/Uni/Masterarbeit/tensorboard"
CHECKPOINT_PATH = "/Users/philipp/Uni/Masterarbeit/checkpoints"


class Retrainer(object):

    def __init__(self, model_def, image_paths):
        self.model_def = model_def
        self.image_paths = image_paths
        self.num_classes = len(image_paths['labels'])
        self.check_paths()


    @staticmethod
    def check_paths():
        """Check if the tensorboard and checkpoint path are existent, otherwise create them"""
        if not os.path.isdir(TENSORBOARD_PATH):
            os.mkdir(TENSORBOARD_PATH)
        if not os.path.isdir(CHECKPOINT_PATH):
            os.mkdir(CHECKPOINT_PATH)


    @staticmethod
    def get_summary_writer(train_vars, loss, accuracy):
        """
        Args:
            train_vars:
            loss:
            accuracy:
        """
        for var in train_vars:
            tf.summary.histogram(var.name, var)

        tf.summary.scalar('cross_entropy', loss)
        tf.summary.scalar('accuracy', accuracy)
        merged_summary = tf.summary.merge_all() # Merge all summaries together
        writer = tf.summary.FileWriter(TENSORBOARD_PATH) # Initialize and return the FileWriter
        return merged_summary, writer


    @staticmethod
    def get_evaluation_op(scores, true_classes):
        """Inserts the operations we need to evaluate the accuracy of our results.

        Args:
            scores: The new final node that produces results
            true_classes: The node we feed the true classes in
        Returns:
            Evaluation operation: defining the accuracy of the model
        """
        with tf.name_scope("accuracy"):
            prediction = tf.argmax(scores, 1)
            correct_pred = tf.equal(prediction, tf.argmax(true_classes, 1))
            accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return accuracy_op


    @staticmethod
    def get_train_op(loss, learning_rate):
        """Inserts the training operation
        Creates an optimizer and applies gradient descent to the trainable variables

        Args:
            loss: the cross entropy mean (scors <> real class)
            var_list: list of all trainable variables
        Returns:
            Traning/optizing operation
        """
        with tf.name_scope("train"):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            # TODO try another optimizer like like tf.train.RMSPropOptimizer(...)
            # see: https://www.tensorflow.org/versions/r0.12/api_docs/python/train/optimizers
            train_op = optimizer.minimize(loss)
            # --> minimize() = combines calls compute_gradients() and apply_gradients()
        return train_op


    @staticmethod
    def get_loss_op(scores, true_classes):
        """Inserts the operations we need to calculate the loss.

        Args:
            scores: The new final node that produces results
            true_classes: The node we feed the true classes in
        Returns: loss operation
        """
        # Op for calculating the loss
        with tf.name_scope("cross_entropy"):
            # softmax_cross_entropy_with_logits 
            # --> calculates the cross entropy between the softmax score (probaility) and hot encoded class expectation (all "0" except one "1") 
            # reduce_mean 
            # --> computes the mean of elements across dimensions of a tensor (cross entropy values here)
            loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=true_classes))
        return loss_op


    def parse_data(self, path, label):
        """

        Args:
            path:
            label:
        """
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)
        # load the image
        image = load_img_as_tensor(
            path,
            self.model_def.input_width,
            self.model_def.input_height,
            crop=False,
            sub_in_mean=self.model_def.subtract_imagenet_mean,
            bgr=self.model_def.use_bgr
        )
        return image, one_hot

    def run(self, finetune_layers, epochs, learning_rate = 0.01, batch_size = 128, keep_prob = 1.0, ckpt_file = ''):
        """
        Run a training on part of the model (retrain/finetune)

        Args:
            root_dir:
            ckpt_file:
            model_def:
        """
        # create datasets
        data_train = Dataset.from_tensor_slices((self.image_paths['training_paths'], self.image_paths['training_labels']))
        data_val  = Dataset.from_tensor_slices((self.image_paths['validation_paths'], self.image_paths['validation_labels']))

        # load and preprocess the images
        data_train = data_train.map(self.parse_data)
        data_val   = data_val.map(self.parse_data)

        # create a new dataset with batches of images
        data_train = data_train.batch(batch_size)
        data_val   = data_val.batch(batch_size)

        # create an reinitializable iterator given the dataset structure
        iterator = Iterator.from_structure(data_train.output_types, data_train.output_shapes)
        next_batch = iterator.get_next()

        # Ops for initializing the two different iterators
        init_op_train = iterator.make_initializer(data_train)
        init_op_val   = iterator.make_initializer(data_val)

        # TF placeholder for graph input and output
        ph_in = tf.placeholder(tf.float32, [batch_size, self.model_def.input_width, self.model_def.input_height, 3])
        ph_out = tf.placeholder(tf.float32, [batch_size, self.num_classes])
        ph_keep_prob = tf.placeholder(tf.float32)

        # Initialize model
        model = self.model_def(ph_in, keep_prob=ph_keep_prob, num_classes=self.num_classes, retrain_layer=finetune_layers)
        
        # Link a variable to model output and get a list of all trainable model-variables
        scores = model.final
        
        # Get a list with all trainable variables
        train_vars = tf.trainable_variables()
        print "=> Will train:"
        for var in train_vars:
            print("  => {}".format(var))
        print "=> Learningrate: %.4f" %learning_rate
        print "=> Batchsize: %i" %batch_size
        print "=> Dropout: %.4f" %(1.0 - keep_prob)
        print "##################################"

        # Add/Get the different operations to optimize (get the loss, train and validate)
        loss_op = self.get_loss_op(scores, ph_out)
        train_op = self.get_train_op(loss_op, learning_rate)
        accuracy_op = self.get_evaluation_op(scores, ph_out)

        # Create a summery and writter to save it to disc
        if WRITE_SUMMARY:
            merged_summary, writer = self.get_summary_writer(train_vars, loss_op, accuracy_op)
        
        # Initialize a saver to store model checkpoints
        if WRITE_CHECKPOINTS:
            saver = tf.train.Saver()

        # Get the number of training/validation steps per epoch to get through all images
        batches_per_epoch_train = int(math.floor(self.image_paths['training_image_count'] / (batch_size + 0.0)))
        batches_per_epoch_val   = int(math.floor(self.image_paths['validation_image_count'] / (batch_size + 0.0)))

        # Start Tensorflow session
        with tf.Session() as sess:
            # Init all variables 
            sess.run(tf.global_variables_initializer())
            # Add the model graph to TensorBoard and print infos
            if WRITE_SUMMARY:
                writer.add_graph(sess.graph)
            # load the pretrained variables or a checkpoint
            if ckpt_file:
                saver.restore(sess, ckpt_file)
            else:
                model.load_initial_weights(sess)

            print("{} Start training...".format(datetime.now()))

            for epoch in range(epochs):
                # ############## TRAINING STEP ##############
                print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
                sess.run(init_op_train)

                for batch_step in range(batches_per_epoch_train):
                    # get next batch of data and run the training operation
                    img_batch, label_batch = sess.run(next_batch)
                    sess.run(train_op, feed_dict={ph_in: img_batch, ph_out: label_batch, ph_keep_prob: keep_prob})

                    if WRITE_SUMMARY and batch_step % DISPLAY_STEP == 0:
                        # Generate summary with the current batch of data and write to file
                        summary = sess.run(merged_summary, feed_dict={ph_in: img_batch, ph_out: label_batch, ph_keep_prob: 1.})
                        writer.add_summary(summary, epoch * batches_per_epoch_train + batch_step)
        
                # ############## VALIDATION STEP ##############
                print("{} Start validation".format(datetime.now()))
                sess.run(init_op_val)
                test_acc = 0.
                test_count = 0

                for _ in range(batches_per_epoch_val):
                    img_batch, label_batch = sess.run(next_batch)
                    acc = sess.run(accuracy_op, feed_dict={ph_in: img_batch, ph_out: label_batch, ph_keep_prob: 1.})
                    test_acc += acc
                    test_count += 1
                    
                test_acc /= test_count
                print("{} Validation Accuracy = {:.10f}".format(datetime.now(), test_acc))
                # tf.logging.info('%s: Validation Accuracy = %.10f%%' % (datetime.now(), test_acc))

                # save checkpoint of the model
                if WRITE_CHECKPOINTS:
                    checkpoint = os.path.join(CHECKPOINT_PATH, 'model_epoch' + str(epoch+1) + '.ckpt')
                    save_path = saver.save(sess, checkpoint)
                    print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint))

            print("{} End training...".format(datetime.now()))