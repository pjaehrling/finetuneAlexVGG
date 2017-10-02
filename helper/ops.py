import tensorflow as tf
from tensorflow.contrib.data import Iterator


def get_validation_ops(scores, true_classes):
    """Inserts the operations we need to evaluate the accuracy of our results.

    Args:
        scores: The new final node that produces results
        true_classes: The node we feed the true classes in
    Returns:
        Evaluation operation: defining the accuracy of the model
    """
    with tf.name_scope("accuracy"):
        predicted_index = tf.argmax(scores, 1)
        true_index = tf.argmax(true_classes, 1)
        correct_pred = tf.equal(predicted_index, true_index)
        accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy_op, predicted_index


def get_train_op(loss, learning_rate, train_vars):
    """Inserts the training operation
    Creates an optimizer and applies gradient descent to the trainable variables

    Args:
        loss: the cross entropy mean (scors <> real class)
        train_vars: list of all trainable variables
    Returns:
        Traning/optizing operation
    """
    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # TODO try another optimizer like like tf.train.RMSPropOptimizer(...)
        # see: https://www.tensorflow.org/versions/r0.12/api_docs/python/train/optimizers
        train_op = optimizer.minimize(loss, var_list=train_vars)
        # --> minimize() = combines calls compute_gradients() and apply_gradients()
    return train_op


def get_loss_op(scores, true_classes):
    """Inserts the operations which calculates the loss.

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


def get_summary_writer_op(train_vars, loss, accuracy, path):
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
    writer = tf.summary.FileWriter(path) # Initialize and return the FileWriter
    return merged_summary, writer


def get_dataset_ops(data_train, data_val, batch_size):
    """
    """
    # create a new dataset with batches of images
    data_train = data_train.batch(batch_size)
    data_val   = data_val.batch(batch_size)

    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(data_train.output_types, data_train.output_shapes)
    next_batch = iterator.get_next()

    # Ops for initializing the two different iterators
    init_op_train = iterator.make_initializer(data_train)
    init_op_val   = iterator.make_initializer(data_val)

    return init_op_train, init_op_val, next_batch