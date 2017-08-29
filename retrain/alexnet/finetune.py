#
# Author: Philipp Jaehrling
# Influenced by: 
# - https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html
# - https://kratzert.github.io/2017/06/15/example-of-tensorflows-new-input-pipeline.html
#

import os
import argparse
import math
import tensorflow as tf
import imageloader as imgl

from alexnet import AlexNet
from datetime import datetime
from tensorflow.python.platform import gfile
from tensorflow.contrib.data import Dataset, Iterator
from tensorflow.python.framework.ops import convert_to_tensor


# Input params
VALIDATION_RATIO = 5 # every 5th element = 1/5 = 0.2 = 20% 
INPUT_WIDTH = 227
INPUT_HEIGHT = 227

# Learning params
LEARNING_RATE = 0.1 # [0.01, 0.1]
NUM_EPOCHS = 2
BATCH_SIZE = 128

# Network params
DROPOUT_RATE = 0.2 # [0.5, 0.2]
FINETUNE_LAYERS = ['fc8', 'fc7', 'fc6']

# Tensorflow/board params
DISPLAY_STEP = 20 # How often to write the tf.summary
TENSORBOARD_PATH = "/tmp/tensorboard"
CHECKPOINT_PATH = "/tmp/checkpoints"

# Number of labels/classes, will be set at runtime
NUM_LABELS = 0

def check_paths():
    """Check if the tensorboard and checkpoint path are existent, otherwise create them"""
    if not os.path.isdir(TENSORBOARD_PATH):
        os.mkdir(TENSORBOARD_PATH)
    if not os.path.isdir(CHECKPOINT_PATH):
        os.mkdir(CHECKPOINT_PATH)



def parse_data(path, label):
    """ """
    # convert label number into one-hot-encoding
    one_hot = tf.one_hot(label, NUM_LABELS)
    # load the image
    image = imgl.load_img_as_tensor(path, INPUT_WIDTH, INPUT_HEIGHT, crop=False, use_mean=True, bgr=True)
    return image, one_hot



def create_summary_writer(train_vars, loss, accuracy):
    """ """
    for var in train_vars:
        tf.summary.histogram(var.name, var)

    tf.summary.scalar('cross_entropy', loss)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary = tf.summary.merge_all() # Merge all summaries together
    writer = tf.summary.FileWriter(TENSORBOARD_PATH) # Initialize and return the FileWriter
    return merged_summary, writer


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


def get_train_op(loss, var_list):
    """Inserts the training operation
    Creates an optimizer and applies gradient descent to the trainable variables
    
    Args: 
        loss: the cross entropy mean (scors <> real class)
        var_list: list of all trainable variables
    Returns: 
        Traning/optizing operation
    """
    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)

        # 1) Blog way
        gradients = tf.gradients(loss, var_list) # returns a list of gradients for each trainable var = len(train_vars)
        gradients = zip(gradients, var_list) # create a list of tuples [(g1, tv1), (g2, tv2), ...]
        train_op = optimizer.apply_gradients(grads_and_vars=gradients)

        # 2) TF retrain way
        # train_op = optimizer.minimize(loss)
        # --> minimize() = combines calls compute_gradients() and apply_gradients()
        # TODO: Find out how they make sure it just trains the last n-layers
        # This takes longer, but has the same validation accuracy??????
    return train_op


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


def finetune(root_dir, ckpt_file):
    """ """
    check_paths()
    image_paths = imgl.load_image_paths_by_subfolder(root_dir, VALIDATION_RATIO)
    
    global NUM_LABELS
    NUM_LABELS = len(image_paths['labels'])

    # create datasets
    data_train = Dataset.from_tensor_slices((image_paths['training_paths'], image_paths['training_labels']))
    data_val  = Dataset.from_tensor_slices((image_paths['validation_paths'], image_paths['validation_labels']))

    # load and preprocess the images
    data_train = data_train.map(parse_data)
    data_val   = data_val.map(parse_data)

    # create a new dataset with batches of images
    data_train = data_train.batch(BATCH_SIZE)
    data_val   = data_val.batch(BATCH_SIZE)

    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(data_train.output_types, data_train.output_shapes)
    next_batch = iterator.get_next()

    # Ops for initializing the two different iterators
    init_op_train = iterator.make_initializer(data_train)
    init_op_val   = iterator.make_initializer(data_val)

    # TF placeholder for graph input and output
    ph_in = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_WIDTH, INPUT_HEIGHT, 3])
    ph_out = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_LABELS])
    ph_dropput = tf.placeholder(tf.float32)

    # Initialize model
    model = AlexNet(ph_in, ph_dropput, NUM_LABELS, FINETUNE_LAYERS)
    
    # Link a variable to model output and get a list of all trainable model-variables
    scores = model.fc8
    train_vars = [var for var in tf.trainable_variables() if var.name.split('/')[0] in FINETUNE_LAYERS]
    # tf.trainable_variables() --> returns all variables createdq

    # Add/Get the different operations to optimize (get the loss, train and validate)
    loss_op = get_loss_op(scores, ph_out)
    train_op = get_train_op(loss_op, train_vars)
    accuracy_op = get_evaluation_op(scores, ph_out)

    # Create a summery and writter to save it to disc
    merged_summary, writer = create_summary_writer(train_vars, loss_op, accuracy_op)
    # Initialize a saver to store model checkpoints
    saver = tf.train.Saver()

    # Get the number of training/validation steps per epoch to get through all images
    batches_per_epoch_train = int(math.floor(image_paths['training_image_count'] / (BATCH_SIZE + 0.0)))
    batches_per_epoch_val   = int(math.floor(image_paths['validation_image_count'] / (BATCH_SIZE + 0.0)))

    # Start Tensorflow session
    with tf.Session() as sess:
        # Init all variables 
        sess.run(tf.global_variables_initializer())
        # Add the model graph to TensorBoard and print infos
        writer.add_graph(sess.graph)
        # load the pretrained variables or a checkpoint
        if ckpt_file:
            saver.restore(sess, ckpt_file)
        else:
            model.load_initial_weights(sess)

        print("{} Start training...".format(datetime.now()))

        for epoch in range(NUM_EPOCHS):
            # ############## TRAINING STEP ##############
            print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
            sess.run(init_op_train)

            for batch_step in range(batches_per_epoch_train):
                # get next batch of data and run the training operation
                img_batch, label_batch = sess.run(next_batch)
                sess.run(train_op, feed_dict={ph_in: img_batch, ph_out: label_batch, ph_dropput: DROPOUT_RATE})

                if batch_step % DISPLAY_STEP == 0:
                    # Generate summary with the current batch of data and write to file
                    summary = sess.run(merged_summary, feed_dict={ph_in: img_batch, ph_out: label_batch, ph_dropput: 1.})
                    writer.add_summary(summary, epoch * batches_per_epoch_train + batch_step)
    
            # ############## VALIDATION STEP ##############
            print("{} Start validation".format(datetime.now()))
            sess.run(init_op_val)
            test_acc = 0.
            test_count = 0

            for _ in range(batches_per_epoch_val):
                img_batch, label_batch = sess.run(next_batch)
                acc = sess.run(accuracy_op, feed_dict={ph_in: img_batch, ph_out: label_batch, ph_dropput: 1.})
                test_acc += acc
                test_count += 1
                
            test_acc /= test_count
            print("{} Validation Accuracy = {:.10f}".format(datetime.now(), test_acc))

            # save checkpoint of the model
            checkpoint = os.path.join(CHECKPOINT_PATH, 'model_epoch' + str(epoch+1) + '.ckpt')
            save_path = saver.save(sess, checkpoint)
            print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint))

        print("{} End training...".format(datetime.now()))


def main():
    """ """
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'root_dir',
        help='Folder with trainings/validation images'
    )
    parser.add_argument(
        '-ckpt',
        type=str,
        default='',
        help='Load this checkpoint file to continue training from this point on'
    )

    args = parser.parse_args()
    root_dir = args.root_dir
    ckpt = args.ckpt

    if not gfile.Exists(root_dir):
        print 'Image root directory \'%s\' not found' %root_dir
        return None

    if ckpt and not gfile.Exists(ckpt):
        print 'Could not find checkpoint file: \'%s\'' %ckpt
        return None

    finetune(root_dir, ckpt)

if __name__ == '__main__':
    main()