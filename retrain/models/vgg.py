#
# Author: Philipp Jaehrling philipp.jaehrling@gmail.com)
#
import tensorflow as tf
import numpy as np

#
# VGG 16 definition for Tensorflow
#
# TODO might use a param to init the net as VGG 19 (adds conv3_4, conv4_4, conv5_4)
#
class VGG(object):

    input_width = 224
    input_height = 224
    subtract_imagenet_mean = True
    use_bgr = True

    def __init__(self, tensor, keep_prob, num_classes, skip_layer, weights_path = './weights/vgg16.npy'):
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
        # 1st Layer: Conv -> Conv -> Pool
        # conv(tensor, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding)
        conv1_1 = conv(self.TENSOR, 3, 3, 64, 1, 1, padding='SAME', name='conv1_1')
        conv1_2 = conv(conv1_1    , 3, 3, 64, 1, 1, padding='SAME', name='conv1_2')
        pool1   = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
        
        # 2nd Layer: Conv -> Conv -> Pool
        conv2_1 = conv(pool1  , 3, 3, 128, 1, 1, padding='SAME', name='conv2_1')
        conv2_2 = conv(conv2_1, 3, 3, 128, 1, 1, padding='SAME', name='conv2_2')
        pool2   = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        # 3rd Layer: Conv -> Conv -> Conv -> Pool
        conv3_1 = conv(pool2  , 3, 3, 256, 1, 1, padding='SAME', name='conv3_1')
        conv3_2 = conv(conv3_1, 3, 3, 256, 1, 1, padding='SAME', name='conv3_2')
        conv3_3 = conv(conv3_2, 3, 3, 256, 1, 1, padding='SAME', name='conv3_3')
        pool3   = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

        # 4th Layer: Conv -> Conv -> Conv -> Pool
        conv4_1 = conv(pool3  , 3, 3, 512, 1, 1, padding='SAME', name='conv4_1')
        conv4_2 = conv(conv4_1, 3, 3, 512, 1, 1, padding='SAME', name='conv4_2')
        conv4_3 = conv(conv4_2, 3, 3, 512, 1, 1, padding='SAME', name='conv4_3')
        pool4   = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

        # 5th Layer: Conv -> Conv -> Conv -> Pool
        conv5_1 = conv(pool4  , 3, 3, 512, 1, 1, padding='SAME', name='conv5_1')
        conv5_2 = conv(conv5_1, 3, 3, 512, 1, 1, padding='SAME', name='conv5_2')
        conv5_3 = conv(conv5_2, 3, 3, 512, 1, 1, padding='SAME', name='conv5_3')
        pool5   = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')

        # 6th Layer: FC -> DropOut
        # [1:] cuts away the first element
        pool5_out  = int(np.prod(pool5.get_shape()[1:])) # 7 * 7 * 512 = 25088
        pool5_flat = tf.reshape(pool5, [-1, pool5_out]) # shape=(image count, 7, 7, 512) -> shape=(image count, 25088)
        fc6        = fc(pool5_flat, num_in=pool5_out, num_out=4096, name='fc6', relu=True)
        dropout1   = tf.nn.dropout(fc6, self.KEEP_PROB)

        # 7th Layer: FC
        fc7      = fc(dropout1, num_in=4096, num_out=4096, name='fc7', relu=True)
        dropout2 = tf.nn.dropout(fc7, self.KEEP_PROB)

        # 8th Layer: FC
        fc8 = fc(dropout2, num_in=4096, num_out=self.NUM_CLASSES, name='fc8', relu=False)
        self.final = fc8

    def load_initial_weights(self, session):
        # Load the weights into memory
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()

        # Loop over all layer ops
        for op_name in weights_dict:
            # Check if the layer is one of the layers that should be reinitialized
            if op_name not in self.SKIP_LAYER:
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


############################################################
# HELPER - Layer wrapper
############################################################

#
# Wrapper around the conv-layer
#
def conv(tensor, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME'):
    input_channels = int(tensor.get_shape()[-1])
    strides = [1, stride_y, stride_x, 1]

    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels, num_filters])
        
        # initializer=tf.truncated_normal( # -> Outputs random values from a truncated normal distribution
        #     [filter_height, filter_width, input_channels, num_filters],
        #     name='weights',
        #     dtype=tf.float32,
        #     stddev=1e-1 # -> standard deviation
        # )

        # Add the convolution
        convolution = tf.nn.conv2d(tensor, weights, strides, padding)

        # Add biases
        biases = tf.get_variable('biases', shape=[num_filters])
        # initializer=tf.constant(0.0, shape=[num_filters], dtype=tf.float32)
        out = tf.reshape(tf.nn.bias_add(convolution, biases), convolution.get_shape().as_list())
        # --> reshape([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3]) ==> [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        # out = tf.nn.bias_add(convolution, biases)
        
        # Apply relu function --> computes rectified linear: max(features, 0)
        relu = tf.nn.relu(out, name=scope.name)
        return relu

#
# Wrapper around the fully connected layer
#
def fc(tensor, num_in, num_out, name, relu = True):
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