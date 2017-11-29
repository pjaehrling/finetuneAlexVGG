#
# Author: Philipp Jaehrling philipp.jaehrling@gmail.com)
#
from preprocessing import resize_crop

class Model(object):
    """
    Parent class for multiple CNN model classes
    """
    # These params should be filled for each model
    image_size = 0
    image_prep = resize_crop

    def __init__(self, tensor, keep_prob, num_classes, retrain_layer, weights_path):
        """
        Args:
            tensor: tf.placeholder, for the input images
            keep_prob: tf.placeholder, for the dropout rate
            num_classes: int, number of classes of the new dataset
            retrain_layer: list of strings, names of the layers you want to reinitialize
            weights_path: path string, path to the pretrained weights (numpy or checkpoint)
        """
        self.tensor = tensor
        self.num_classes = num_classes
        self.keep_prob = keep_prob
        self.retrain_layer = retrain_layer
        self.weights_path = weights_path

    def get_final_op(self):
        """Get the net output (final op)
            
        Returns: the last op containing the log predictions and end_points dict
        """
        raise NotImplementedError("Subclass must implement method")

    def get_endpoints(self):
        """Get an ordered dict with all endpoints
            
        Returns: ordered endpoints dict
        """
        raise NotImplementedError("Subclass must implement method")

    def get_restore_vars(self):
        """Get a list of tensors, which should be restored
        """
        raise NotImplementedError("Subclass must implement method")

    def get_retrain_vars(self):
        """Get a list of tensors, which should be retrained
        """
        raise NotImplementedError("Subclass must implement method")

    def load_initial_weights(self, session):
        """Load the initial weights

        Args:
            session: current tensorflow session
        """
        raise NotImplementedError("Subclass must implement method")

    def is_layer_trainable(self, layer_name):
        """Return is a layer is trainable or not
        """
        return True if layer_name in self.retrain_layer else False