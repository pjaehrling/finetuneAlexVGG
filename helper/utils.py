import os
from datetime import datetime  
    
def save_session_to_checkpoint_file(sess, saver, epoch, path):
    """
    """
    checkpoint = os.path.join(path, 'model_epoch' + str(epoch+1) + '.ckpt')
    saver.save(sess, checkpoint)
    print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint))

def get_misclassified(batch_start_index, predictions, paths, labels):
    """
    Returns: a list of tupels (path, predicted label, true label)
    """
    misclassified = []
    for i in range(len(predictions)):
        index = batch_start_index + i
        if predictions[i] != labels[index]:
            misclassified.append((paths[index], predictions[i], labels[index]))

    return misclassified

def print_misclassified(misclassified, labels):
    """
    """
    print("=> Misclassified (%i)" %len(misclassified))
    for _, (img_path, predicted_label, true_label) in enumerate(misclassified):
        print("  => {} -> {} ({})".format(
            img_path,
            labels[predicted_label],
            labels[true_label]
        ))

def run_training(sess, train_op, iterator_op, get_next_batch_op, ph_data, ph_labels, ph_keep_prob, 
                keep_prob, batches, epoch, write_summary = False, summary_op = None, writer = None, disply_step = 1):
    """
    Args:
        sess:
        train_op:
        iterator_op:
        get_next_batch_op:
        ph_data:
        ph_labels:
        ph_keep_prob:
        keep_prob:
        batches:
        epoch:
        summary_op:
        writer:
    """
    print("{} Start training...".format(datetime.now()))
    sess.run(iterator_op)

    for batch_step in range(batches):
        # Get next batch of data and run the training operation
        data_batch, label_batch = sess.run(get_next_batch_op)
        sess.run(
            train_op,
            feed_dict={ph_data: data_batch, ph_labels: label_batch, ph_keep_prob: keep_prob}
        )

        if write_summary and batch_step % disply_step == 0:
            # Generate summary with the current batch of data and write to file
            summary = sess.run(summary_op, feed_dict={ph_data: data_batch, ph_labels: label_batch, ph_keep_prob: 1.})
            writer.add_summary(summary, epoch * batches + batch_step)


def run_validation(sess, accuracy_op, predicted_index_op, iterator_op, get_next_batch_op, ph_data, ph_labels, ph_keep_prob,
                    batches, batch_size, is_last, show_misclassified = False, data = {}):
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
        batch_size:
        epoch:
        is_last:
        show_misclassified:
    """
    # Variables to keep track over different batches
    test_acc = 0.
    test_count = 0
    misclassified = []

    print("{} Start validation...".format(datetime.now()))
    sess.run(iterator_op)

    predicted = 0

    for batch_step in range(batches):
        img_batch, label_batch = sess.run(get_next_batch_op)
        acc, predictions = sess.run(
            [accuracy_op, predicted_index_op],
            feed_dict={ph_data: img_batch, ph_labels: label_batch, ph_keep_prob: 1.}
        )
        test_acc += acc
        test_count += 1
        predicted += len(predictions)
        
        if is_last and show_misclassified:
            start_index =  batch_step * batch_size
            misclassified += get_misclassified(
                start_index,
                predictions,
                data['validation_paths'],
                data['validation_labels']
            )
    
    # Calculate the overall validation accuracy
    test_acc /= test_count
    print("{} Validation Accuracy = {:.10f} ({})".format(datetime.now(), test_acc, predicted))

    if is_last and show_misclassified:
        print_misclassified(misclassified, data['labels'])