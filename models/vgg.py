#
# Author: Philipp Jaehrling philipp.jaehrling@gmail.com)
#
import tensorflow as tf
import numpy as np

from models.model import Model
from preprocessing.imagenet.bgr import resize_crop

class VGG(Model):
    """
    VGG16 model definition for Tensorflow
    """

    image_size = 224
    image_prep = resize_crop

    def __init__(self, tensor, keep_prob=1.0, num_classes=1000, retrain_layer=[], weights_path='./weights/vgg16.npy'):
        # Call the parent class, which will create the graph
        Model.__init__(self, tensor, keep_prob, num_classes, retrain_layer, weights_path)
        
        # Call the create function to build the computational graph
        self.create()

    def get_prediction(self):
        return self.final

    def get_restore_vars(self):
        return [v for v in tf.global_variables() if not v.name.split('/')[0] in self.retrain_layer]

    def get_retrain_vars(self):
        return tf.trainable_variables()

    def load_initial_weights(self, session):
        self.load_initial_numpy_weights(session)

    def create(self):
        # 1st Layer: Conv -> Conv -> Pool
        # conv(tensor, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding)
        conv1_1 = self.conv(self.tensor, 3, 3, 64, 1, 1, padding='SAME', name='conv1_1')
        conv1_2 = self.conv(conv1_1    , 3, 3, 64, 1, 1, padding='SAME', name='conv1_2')
        pool1   = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
        
        # 2nd Layer: Conv -> Conv -> Pool
        conv2_1 = self.conv(pool1  , 3, 3, 128, 1, 1, padding='SAME', name='conv2_1')
        conv2_2 = self.conv(conv2_1, 3, 3, 128, 1, 1, padding='SAME', name='conv2_2')
        pool2   = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        # 3rd Layer: Conv -> Conv -> Conv -> Pool
        conv3_1 = self.conv(pool2  , 3, 3, 256, 1, 1, padding='SAME', name='conv3_1')
        conv3_2 = self.conv(conv3_1, 3, 3, 256, 1, 1, padding='SAME', name='conv3_2')
        conv3_3 = self.conv(conv3_2, 3, 3, 256, 1, 1, padding='SAME', name='conv3_3')
        pool3   = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

        # 4th Layer: Conv -> Conv -> Conv -> Pool
        conv4_1 = self.conv(pool3  , 3, 3, 512, 1, 1, padding='SAME', name='conv4_1')
        conv4_2 = self.conv(conv4_1, 3, 3, 512, 1, 1, padding='SAME', name='conv4_2')
        conv4_3 = self.conv(conv4_2, 3, 3, 512, 1, 1, padding='SAME', name='conv4_3')
        pool4   = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

        # 5th Layer: Conv -> Conv -> Conv -> Pool
        conv5_1 = self.conv(pool4  , 3, 3, 512, 1, 1, padding='SAME', name='conv5_1')
        conv5_2 = self.conv(conv5_1, 3, 3, 512, 1, 1, padding='SAME', name='conv5_2')
        conv5_3 = self.conv(conv5_2, 3, 3, 512, 1, 1, padding='SAME', name='conv5_3')
        pool5   = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')

        # 6th Layer: FC -> DropOut
        # [1:] cuts away the first element
        pool5_out  = int(np.prod(pool5.get_shape()[1:])) # 7 * 7 * 512 = 25088
        pool5_flat = tf.reshape(pool5, [-1, pool5_out]) # shape=(image count, 7, 7, 512) -> shape=(image count, 25088)
        fc6        = self.fc(pool5_flat, num_in=pool5_out, num_out=4096, name='fc6', relu=True)
        dropout1   = tf.nn.dropout(fc6, self.keep_prob)

        # 7th Layer: FC
        fc7      = self.fc(dropout1, num_in=4096, num_out=4096, name='fc7', relu=True)
        dropout2 = tf.nn.dropout(fc7, self.keep_prob)

        # 8th Layer: FC
        fc8 = self.fc(dropout2, num_in=4096, num_out=self.num_classes, name='fc8', relu=False)
        self.final = fc8
