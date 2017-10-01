#
# Author: Philipp Jaehrling philipp.jaehrling@gmail.com)
#
from models.model import Model
from preprocessing.imagenet import resize_crop

import tensorflow.contrib.slim as slim
from tensorflow import trainable_variables
from tensorflow.contrib.slim.python.slim.nets.vgg import vgg_16
from tensorflow.contrib.slim.python.slim.nets.vgg import vgg_arg_scope

class VGGslim(Model):
    """
    VGG16 model definition for Tensorflow.
    All the "fully connected" layers have been transformed to "conv2d" layers in this implementation.
    """
    image_size = vgg_16.default_image_size
    image_prep = resize_crop

    def __init__(self, tensor, keep_prob=1.0, num_classes=1000, retrain_layer=[], weights_path='./weights/vgg_16.ckpt'):
        # Call the parent class
        Model.__init__(self, tensor, keep_prob, num_classes, retrain_layer, weights_path)

        # When no layer is retrained it's not training
        # TODO could this be a problem in validation while training?
        self.is_training = True if retrain_layer else False

    def get_prediction(self):
        with slim.arg_scope(vgg_arg_scope()):
            predictions, _ = vgg_16(self.tensor, num_classes=self.num_classes, is_training=self.is_training)
            return predictions

    def get_restore_vars(self):
        return [v for v in slim.get_variables_to_restore() if not v.name.split('/')[1] in self.retrain_layer]

    def get_retrain_vars(self):
        return [v for v in trainable_variables() if v.name.split('/')[1] in self.retrain_layer]

    def load_initial_weights(self, session):
        self.load_initial_checkpoint(session)
