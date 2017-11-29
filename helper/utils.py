import os
from datetime import datetime
from random import randint  
from tensorflow.python.ops.nn_ops import softmax
    
def save_session_to_checkpoint_file(sess, saver, epoch, path):
    """
    """
    checkpoint = os.path.join(path, datetime.now().strftime("%m%d_%H%M%S_") + 'model_epoch' + str(epoch+1) + '.ckpt')
    saver.save(sess, checkpoint)
    print("Model checkpoint saved at {}".format(checkpoint))

def get_misclassified(corr_pred, paths, pred_index, true_index, scores):
    """
    Returns: a list of tupels (path, predicted label, true label)
    """
    misclassified = []
    for i, correct in enumerate(corr_pred):
        if not correct:
            misclassified.append((paths[i], pred_index[i], true_index[i], scores[i]))

    return misclassified

def print_misclassified(sess, misclassified, labels):
    """
    """
    print("----------------------------------------------------------------")
    print("")
    print("=> Misclassified: %i" %len(misclassified))
    print("================================================================")
    for (path, pred_index, true_index, score) in misclassified:
        smax = sess.run(softmax(score))

        print("{} | {} ({}) | {}".format(
            path,
            labels[pred_index],
            labels[true_index],
            smax
        ))
    print("================================================================")

def print_output_header(train_count, val_count):
    """
    """
    print("=> Getting loss and accuracy for:")
    print("  => {} training entries".format(train_count))
    print("  => {} validation entries".format(val_count))
    print("")
    print(" Ep  |   Time   |   T Loss   |   V Loss   |  T Accu. |  V Accu.")
    print("----------------------------------------------------------------")

def print_output_epoch(epoch, train_loss, train_acc, test_loss, test_acc):
    """
    """
    print("{:4.0f} | {} | {:.8f} | {:.8f} | {:.6f} | {:.6f}".format(
        epoch,
        datetime.now().strftime("%H:%M:%S"),
        train_loss,
        test_loss,
        train_acc,
        test_acc
    ))

def run_training(sess, train_op, loss_op, accuracy_op, iterator_op, get_next_batch_op, ph_data, ph_labels, ph_keep_prob, keep_prob, batches):
    """
    Args:
        sess:
        loss_op,
        train_op:
        accuracy_op:
        iterator_op:
        get_next_batch_op:
        ph_data:
        ph_labels:
        ph_keep_prob:
        keep_prob:
        batches:
    """
    # Variables to keep track over different batches
    acc = 0.
    loss = 0.
    # use_batch_for_crossvalidation = randint(0, batches - 2)
    # -2 -> -1 = we start at index 0 / -1 we don't want to use the last batch, it might be smaller

    sess.run(iterator_op)
    for batch_step in range(batches):
        # Get next batch of data and run the training operation
        data_batch, label_batch, _ = sess.run(get_next_batch_op)
        _, batch_loss, batch_acc = sess.run(
            [train_op, loss_op, accuracy_op],
            feed_dict={ph_data: data_batch, ph_labels: label_batch, ph_keep_prob: keep_prob}
        )
        loss += batch_loss
        acc += batch_acc

    acc /= batches
    loss /= batches
    return loss, acc


def run_validation(sess, loss_op, accuracy_op, correct_prediction_op, predicted_index_op, true_index_op, final_op,
                   iterator_op, get_next_batch_op, ph_data, ph_labels, ph_keep_prob, batches, return_misclassified = False):
    """
    Args:
        sess:
        accuracy_op:
        predicted_index_op:
        iterator_op:
        get_next_batch_op:
        ph_data:
        ph_labels:
        ph_keep_prob:
        batches:
        return_misclassified:
        data
    """
    # Variables to keep track over different batches
    acc = 0.
    loss = 0.
    misclassified = []

    sess.run(iterator_op)
    for _ in range(batches):
        img_batch, label_batch, paths = sess.run(get_next_batch_op)
        scores, batch_loss, batch_acc, corr_pred, pred_index, true_index = sess.run(
            [final_op, loss_op, accuracy_op, correct_prediction_op, predicted_index_op, true_index_op],
            feed_dict={ph_data: img_batch, ph_labels: label_batch, ph_keep_prob: 1.}
        )
        loss += batch_loss
        acc += batch_acc
        
        if return_misclassified:
            misclassified += get_misclassified(corr_pred, paths, pred_index, true_index, scores)
    
    acc /= batches
    loss /= batches
    return loss, acc, misclassified