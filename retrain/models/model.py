#
# Author: Philipp Jaehrling philipp.jaehrling@gmail.com)
# Highly influenced by: https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html
#
import tensorflow as tf
import numpy as np

#
# Parent class for models
#
class Model(object):
    """
    Parent class for multiple CNN model classes
    """

    # These params should be filled for each model
    input_width = 0
    input_height = 0
    subtract_imagenet_mean = True
    use_bgr = True


    def __init__(self, tensor, keep_prob, num_classes, retrain_layer, weights_path):
        """
        Args:
            tensor: tf.placeholder, for the input images
            keep_prob: tf.placeholder, for the dropout rate
            num_classes: int, number of classes of the new dataset
            retrain_layer: list of strings, names of the layers you want to reinitialize
            weights_path: path string, path to the pretrained weights
        """
   
        # Parse input arguments
        self.tensor = tensor
        self.num_classes = num_classes
        self.keep_prob = keep_prob
        self.retrain_layer = retrain_layer
        self.weights_path = weights_path
        
        # Set output to be input be default, will be set as soon as we create the graph
        self.final = tensor 

        # Call the create function to build the computational graph
        self.create()


    def create(self):
        """
        Build/Create the model graph
        """
        raise NotImplementedError("Subclass must implement abstract method")


    def load_initial_weights(self, session):
        """
        Load the initial weights
        Do not init the layers that we want to retrain

        Args:
            session: current tensorflow session
        """
        # Load the weights into memory
        weights_dict = np.load(self.weights_path, encoding='bytes').item()

        # Loop over all layer ops
        for op_name in weights_dict:
            # Check if the layer is one of the layers that should be reinitialized
            if op_name not in self.retrain_layer:
                with tf.variable_scope(op_name, reuse=True):
                    # Loop over list of weights/biases and assign them to their corresponding tf variable
                    for data in weights_dict[op_name]:
                        # Biases
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable = False)
                            session.run(var.assign(data)) 
                        # Weights
                        else:
                            var = tf.get_variable('weights', trainable = False)
                            session.run(var.assign(data))


    @staticmethod
    def conv(tensor, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME', groups=1):
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
        channels_per_layer = input_channels / groups # In case we split the data for multiple parallel conv-layer
        strides = [1, stride_y, stride_x, 1]
        
        # Create lambda function for the convolution
        convolve = lambda input, kernel: tf.nn.conv2d(input, kernel, strides=strides, padding=padding)
        # tf.nn.conv2d --> Computes a 2-D convolution given 4-D input and filter tensors

        with tf.variable_scope(name) as scope:
            # tf.get_variable(...) --> get an existing variable with these parameters or create a new one
            # ... prefixes the name with the current variable scope and performs reuse checks
            weights = tf.get_variable('weights', shape=[filter_height, filter_width, channels_per_layer, num_filters])
            # initializer=tf.truncated_normal( # -> Outputs random values from a truncated normal distribution
            #     [filter_height, filter_width, channels_per_layer, num_filters],
            #     name='weights',
            #     dtype=tf.float32,
            #     stddev=1e-1 # -> standard deviation
            # )

            # Add the convolution
            if groups == 1:
                convolution = convolve(tensor, weights)
            else:
                # In the cases of multiple groups, split inputs & weights and convolve them separately
                input_groups  = tf.split(num_or_size_splits=groups, value=tensor, axis=3)
                weight_groups = tf.split(num_or_size_splits=groups, value=weights, axis=3)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]
                # Concat the convolved output together again
                convolution = tf.concat(values=output_groups, axis=3)

            # Add biases
            biases = tf.get_variable('biases', shape=[num_filters])
            # initializer=tf.constant(0.0, shape=[num_filters], dtype=tf.float32)
            out = tf.reshape(tf.nn.bias_add(convolution, biases), convolution.get_shape().as_list())
            # --> reshape([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3]) ==> [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
            # out = tf.nn.bias_add(convolution, biases)
            
            # Apply relu function --> computes rectified linear: max(features, 0)
            relu = tf.nn.relu(out, name=scope.name)
            return relu


    @staticmethod
    def fc(tensor, num_in, num_out, name, relu = True):
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
        with tf.variable_scope(name) as scope:

            # Create tf variables for the weights and biases
            weights = tf.get_variable('weights', shape=[num_in, num_out])
            # initializer=tf.truncated_normal([num_in, num_out], dtype=tf.float32, stddev=1e-1)
            biases = tf.get_variable('biases', [num_out])
            # initializer=tf.constant(1.0, shape=[num_out], dtype=tf.float32)

            # Matrix multiply weights and inputs and add bias
            # act = tf.nn.bias_add(tf.matmul(tensor, weights), biases, name=scope)
            act = tf.nn.xw_plus_b(tensor, weights, biases, name=scope.name)

            if relu is True:
                # Apply relu function --> computes rectified linear: max(features, 0)
                relu = tf.nn.relu(act, name=scope.name)
                return relu
            else:
                return act
