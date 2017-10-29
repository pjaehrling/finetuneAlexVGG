#
# Author: Philipp Jaehrling philipp.jaehrling@gmail.com)
#
from models.model import Model
from preprocessing import inception as inception_prepocessing
# from preprocessing.imagenet.bgr import resize_crop
from weight_loading.checkpoint import load_weights

import tensorflow.contrib.slim as slim
from tensorflow import trainable_variables
from tensorflow.contrib.slim.python.slim.nets.resnet_v2 import resnet_v2
from tensorflow.contrib.slim.python.slim.nets.resnet_v2 import resnet_v2_101
from tensorflow.contrib.slim.python.slim.nets.resnet_v2 import resnet_arg_scope

class ResNetV2(Model):
    """
    ResNet V2 model wrapper for the Tensorflow Slim implementation.
    All the "fully connected" layers have been transformed to "conv2d" layers in this implementation.
    """
    image_size = resnet_v2.default_image_size
    image_prep = inception_prepocessing

    def __init__(self, tensor, keep_prob=1.0, num_classes=1001, retrain_layer=[], weights_path='./weights/resnet_v2_101.ckpt'):
        # Call the parent class
        Model.__init__(self, tensor, keep_prob, num_classes, retrain_layer, weights_path)

        # TODO This implementation has a problem while validation (is still set to training)
        is_training = True if retrain_layer else False
        with slim.arg_scope(resnet_arg_scope()):
            self.final, self.endpoints = resnet_v2_101(
                self.tensor,
                num_classes=num_classes,
                is_training=is_training
            )

    def get_final_op(self):
        return self.final

    def get_endpoints(self):
        return self.endpoints

    def get_restore_vars(self):
        return [v for v in slim.get_variables_to_restore() if not v.name.split('/')[1] in self.retrain_layer]

    def get_retrain_vars(self):
        return [v for v in trainable_variables() if v.name.split('/')[1] in self.retrain_layer]

    def load_initial_weights(self, session):
        load_weights(session, self.weights_path, self.get_restore_vars())