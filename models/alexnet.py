#
# Author: Philipp Jaehrling philipp.jaehrling@gmail.com)
# Influenced by: https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html
#
from collections import OrderedDict

import tensorflow as tf
import numpy as np

from models.model import Model
from preprocessing.imagenet.bgr import resize_crop
from weight_loading.numpyfile import load_weights

from helper.layer import fc, conv

class AlexNet(Model):
    """
    AlexNet model definition for Tensorflow
    """
    image_size = 227
    image_prep = resize_crop

    def __init__(self, tensor, keep_prob=1.0, num_classes=1000, retrain_layer=[], weights_path='./weights/bvlc_alexnet.npy'):
        # Call the parent class, which will create the graph
        Model.__init__(self, tensor, keep_prob, num_classes, retrain_layer, weights_path)

        # Call the create function to build the computational graph
        self.final, self.endpoints = self.create()

    def get_final_op(self):
        return self.final

    def get_endpoints(self):
        return self.endpoints

    def get_restore_vars(self):
        return [v for v in tf.global_variables() if not v.name.split('/')[0] in self.retrain_layer]

    def get_retrain_vars(self):
        return tf.trainable_variables()

    def load_initial_weights(self, session):
        load_weights(session, self.weights_path, self.retrain_layer)

    def create(self):
        # 1st Layer: Conv (w ReLu) -> Pool -> Lrn
        conv1 = conv(self.tensor, 11, 11, 96, 4, 4, padding='VALID', name='conv1', trainable=self.is_layer_trainable('conv1'))        
        norm1 = tf.nn.local_response_normalization(conv1, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0, name='norm1')
        pool1 = tf.nn.max_pool(norm1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
    
        # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
        conv2 = conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2', trainable=self.is_layer_trainable('conv2'))
        norm2 = tf.nn.local_response_normalization(conv2, depth_radius=2,alpha=2e-05, beta=0.75, bias=1.0, name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
        
        # 3rd Layer: Conv (w ReLu)
        conv3 = conv(pool2, 3, 3, 384, 1, 1, name='conv3', trainable=self.is_layer_trainable('conv3'))
        
        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4', trainable=self.is_layer_trainable('conv4'))
        
        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5', trainable=self.is_layer_trainable('conv5'))
        pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
        
        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        pool5_out  = int(np.prod(pool5.get_shape()[1:])) # 6 * 6 * 256 = 9216
        pool5_flat = tf.reshape(pool5, [-1, pool5_out]) # shape=(image count, 6, 6, 256) -> shape=(image count, 9216)
        fc6        = fc(pool5_flat, 4096, name='fc6', trainable=self.is_layer_trainable('fc6'))
        dropout6   = tf.nn.dropout(fc6, self.keep_prob)
        
        # 7th Layer: FC (w ReLu) -> Dropout
        fc7      = fc(dropout6, 4096, name='fc7', trainable=self.is_layer_trainable('fc7'))
        dropout7 = tf.nn.dropout(fc7, self.keep_prob)
    
        # 8th Layer: FC and return unscaled activations (for tf.nn.softmax_cross_entropy_with_logits)
        fc8 = fc(dropout7, self.num_classes, relu=False, name='fc8', trainable=self.is_layer_trainable('fc8'))

        # add layers to the endpoints dict
        endpoints = OrderedDict()
        endpoints['conv1'] = conv1
        endpoints['pool1'] = pool1
        endpoints['conv2'] = conv2
        endpoints['pool2'] = pool2
        endpoints['conv3'] = conv3
        endpoints['conv4'] = conv4
        endpoints['conv5'] = conv5
        endpoints['pool5'] = pool5
        endpoints['pool5/flat'] = pool5_flat # 9216
        endpoints['fc6'] = fc6 # 4096
        endpoints['fc7'] = fc7 # 4096
        endpoints['fc8'] = fc8 # number of classes

        return fc8, endpoints

