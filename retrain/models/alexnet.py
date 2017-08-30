#
# Author: Philipp Jaehrling philipp.jaehrling@gmail.com)
# Highly influenced by: https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html
#
import tensorflow as tf
import numpy as np

#
# AlexNet definition for Tensorflow
#
class AlexNet(object):

    input_width = 227
    input_height = 227
    subtract_imagenet_mean = True
    use_bgr = True

    def __init__(self, tensor, keep_prob, num_classes, skip_layer, weights_path = './weights/bvlc_alexnet.npy'):
        # - tensor: tf.placeholder, for the input images
        # - keep_prob: tf.placeholder, for the dropout rate
        # - num_classes: int, number of classes of the new dataset
        # - skip_layer: list of strings, names of the layers you want to reinitialize
        # - weights_path: path string, path to the pretrained weights,
   
        # Parse input arguments
        self.TENSOR = tensor
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.SKIP_LAYER = skip_layer
        self.WEIGHTS_PATH = weights_path

        # Call the create function to build the computational graph of AlexNet
        self.create()

    def create(self):
        # 1st Layer: Conv (w ReLu) -> Pool -> Lrn
        conv1 = conv(self.TENSOR, 11, 11, 96, 4, 4, padding = 'VALID', name = 'conv1')
        pool1 = max_pool(conv1, 3, 3, 2, 2, padding = 'VALID', name = 'pool1')
        norm1 = lrn(pool1, 2, 2e-05, 0.75, name = 'norm1')
    
        # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
        conv2 = conv(norm1, 5, 5, 256, 1, 1, groups = 2, name = 'conv2')
        pool2 = max_pool(conv2, 3, 3, 2, 2, padding = 'VALID', name ='pool2')
        norm2 = lrn(pool2, 2, 2e-05, 0.75, name = 'norm2')
        
        # 3rd Layer: Conv (w ReLu)
        conv3 = conv(norm2, 3, 3, 384, 1, 1, name = 'conv3')
        
        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = conv(conv3, 3, 3, 384, 1, 1, groups = 2, name = 'conv4')
        
        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = conv(conv4, 3, 3, 256, 1, 1, groups = 2, name = 'conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding = 'VALID', name = 'pool5')
        
        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(pool5, [-1, 6*6*256])
        fc6 = fc(flattened, 6*6*256, 4096, name='fc6')
        dropout6 = dropout(fc6, self.KEEP_PROB)
        
        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fc(dropout6, 4096, 4096, name = 'fc7')
        dropout7 = dropout(fc7, self.KEEP_PROB)
    
        # 8th Layer: FC and return unscaled activations (for tf.nn.softmax_cross_entropy_with_logits)
        self.final = fc(dropout7, 4096, self.NUM_CLASSES, relu = False, name='fc8')

    def load_initial_weights(self, session):
        # 1. weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/ come as a dict of lists (e.g. weights['conv1'] is a list)
        # 2. weights converted with caffe-to-tensorflow come as dict of dicts (e.g. weights['conv1'] is another dict with the keys weights and biases)
        # At least for the first case we need a special load function
    
        # Load the weights into memory
        weights_dict = np.load(self.WEIGHTS_PATH, encoding = 'bytes').item()
    
        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:
        
            # Check if the layer is one of the layers that should be reinitialized
            if op_name not in self.SKIP_LAYER:
                with tf.variable_scope(op_name, reuse = True):
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


############################################################
# HELPER - Layer wrapper
############################################################

#
# Wrapper around the conv-layer
# We need this because AlexNet splits up channels for different parallel conv layers
#
def conv(tensor, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME', groups=1):

    # Get number of input channels
    input_channels = int(tensor.get_shape()[-1])
    # In TensorFlow, a tensor has both a static (inferred) shape and a dynamic (true) shape.
    # tf.Tensor.get_shape(t) --> static shape
    # tf.shape(t) --> dynamic shape (If the static shape is not fully define)

    # Create lambda function for the convolution
    strides = [1, stride_y, stride_x, 1]
    convolve = lambda input, filter: tf.nn.conv2d(input, filter, strides, padding)
    # tf.nn.conv2d --> Computes a 2-D convolution given 4-D input and filter tensors

    # tf.variable_scope(name) --> returns a context manager for defining ops that create variables (layers)
    # - validates that the (optional) values are from the same graph
    # - ensures that graph is the default graph
    with tf.variable_scope(name) as scope:
        # In case we split the data for multiple parallel conv-layer
        channels_per_layer = input_channels / groups

        # tf.get_variable(...) --> get an existing variable with these parameters or create a new one
        weights = tf.get_variable('weights', shape=[filter_height, filter_width, channels_per_layer, num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

        if groups == 1:
            conv = convolve(tensor, weights)
        else:
            # In the cases of multiple groups, split inputs & weights and convolve them separately
            # axis = dimension along which to split
            input_groups = tf.split(num_or_size_splits=groups, value=tensor, axis=3)
            weight_groups = tf.split(num_or_size_splits=groups, value=weights, axis=3)
            output_groups = [convolve(input, filter) for input, filter in zip(input_groups, weight_groups)]

            # Concat the convolved output together again
            conv = tf.concat(values=output_groups, axis=3)

        # Add biases
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        # tf.reshape example:
        # tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9] ... has shape [9]
        # reshape(t, [3, 3]) ==> [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        # Apply relu function --> computes rectified linear: max(features, 0)
        relu = tf.nn.relu(bias, name=scope.name)

        return relu


#
# Wrapper around the fully connected layer
#
def fc(tensor, num_in, num_out, name, relu = True):
    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(tensor, weights, biases, name=scope.name)

        if relu == True:
            # Apply ReLu non linearity
            relu = tf.nn.relu(act)
            return relu
        else:
            return act

#
# Wrapper around Max-Pooling
#
def max_pool(tensor, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    return tf.nn.max_pool(tensor, ksize=[1, filter_height, filter_width, 1], strides=[1, stride_y, stride_x, 1], padding=padding, name=name)

#
# Wrapper around Local-Response-Normalization
#
def lrn(tensor, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(tensor, depth_radius = radius,alpha=alpha, beta=beta, bias=bias, name=name)

#
# Wrapper around dropout
#
def dropout(tensor, keep_prob):
    return tf.nn.dropout(tensor, keep_prob)
