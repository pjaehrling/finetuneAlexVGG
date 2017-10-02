#
# Author: Philipp Jaehrling philipp.jaehrling@gmail.com)
# Influenced by: https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html
#
import tensorflow as tf

#
# Parent class for models
#
class Model(object):
    """
    Parent class for multiple CNN model classes
    """
    # These params should be filled for each model
    image_size = 0
    image_prep = False

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

    ############################################################################
    # LAYER HELPER FUNCTIONS
    ############################################################################
    def conv(self, tensor, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME', groups=1):
        """
        Wrapper around the tensorflow conv-layer op

        Args:
            tensor:
            filter_height:
            filter_width:
            num_filters:
            stride_y:
            stride_x:
            name:
            padding:
            groups:

        Returns:
        """
        input_channels = int(tensor.get_shape()[-1])
        channels_per_layer = int(input_channels / groups) # In case we split the data for multiple parallel conv-layer
        strides = [1, stride_y, stride_x, 1]
        shape = [filter_height, filter_width, channels_per_layer, num_filters]
        trainable = True if name in self.retrain_layer else False
        
        # -> Outputs random values from a truncated normal distribution (with the given standard deviation)
        init_w = tf.truncated_normal(shape, name='weights', dtype=tf.float32, stddev=0.001)
        init_b = tf.zeros([num_filters])
        
        # tf.nn.conv2d --> Computes a 2-D convolution given 4-D input and filter tensors
        convolve = lambda input, kernel: tf.nn.conv2d(input, kernel, strides=strides, padding=padding)

        with tf.variable_scope(name) as scope:
            # tf.get_variable(...) --> get an existing variable with these parameters or create a new one
            # ... prefixes the name with the current variable scope and performs reuse checks
            weights = tf.get_variable(
                'weights',
                # shape=shape,
                trainable=trainable,
                initializer=init_w
            )

            # Add the convolution
            if groups == 1:
                convolution = convolve(tensor, weights)
            else:
                # In the cases of multiple groups, split inputs & weights and convolve them separately
                input_groups  = tf.split(num_or_size_splits=groups, value=tensor, axis=3)
                weight_groups = tf.split(num_or_size_splits=groups, value=weights, axis=3)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]
                convolution = tf.concat(values=output_groups, axis=3)

            # Add biases
            biases = tf.get_variable(
                'biases',
                # shape=[num_filters],
                trainable=trainable,
                initializer=init_b
            )
            
            # out = tf.reshape(tf.nn.bias_add(convolution, biases), convolution.get_shape().as_list())
            # --> reshape([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3]) ==> [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
            out = tf.nn.bias_add(convolution, biases)
            
            # Apply relu function --> computes rectified linear: max(features, 0)
            relu = tf.nn.relu(out, name=scope.name)
            return relu


    def fc(self, tensor, num_in, num_out, name, relu = True):
        """
        Wrapper around the tensorflow fully connected layer op

        Args:
            tensor:
            num_in:
            num_out:
            name:
            relu:

        Returns:
        """
        trainable = True if name in self.retrain_layer else False
        init_w = tf.truncated_normal([num_in, num_out], name='weights', dtype=tf.float32, stddev=0.001)
        # TODO: try other ways to initializ
        # see: https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow#36784797
        init_b = tf.ones([num_out]) # TODO <- zeros gave no training increase at all

        with tf.variable_scope(name) as scope:

            # Create tf variables for the weights and biases
            weights = tf.get_variable(
                'weights',
                # shape=[num_in, num_out],
                trainable=trainable,
                initializer=init_w
            )
            biases = tf.get_variable(
                'biases',
                # shape=[num_out],
                trainable=trainable,
                initializer=init_b
            )

            # Matrix multiply weights and inputs and add bias
            # act = tf.nn.bias_add(tf.matmul(tensor, weights), biases, name=scope)
            act = tf.nn.xw_plus_b(tensor, weights, biases, name=scope.name)

            if relu is True:
                # Apply relu function --> computes rectified linear: max(features, 0)
                relu = tf.nn.relu(act, name=scope.name)
                return relu
            else:
                return act
