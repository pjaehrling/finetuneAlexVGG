#
# Author: Philipp Jaehrling philipp.jaehrling@gmail.com)
#
from models.model import Model
from preprocessing import vgg_preprocessing

import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets.vgg import vgg_16
from tensorflow.contrib.slim.python.slim.nets.vgg import vgg_arg_scope

class VGGslim(Model):
    """
    VGG16 model definition for Tensorflow
    """
    image_size = vgg_16.default_image_size
    image_prep = vgg_preprocessing

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

    def load_initial_weights(self, session):
        self.load_initial_checkpoint(session)
