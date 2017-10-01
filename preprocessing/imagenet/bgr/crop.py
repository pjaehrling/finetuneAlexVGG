import tensorflow as tf
from preprocessing.imagenet import crop

def preprocess_image(image, output_height, output_width, is_training=False):
    img_standardized = crop.preprocess_image(image, output_height, output_width, is_training)

    # e.g. in my alexnet implementation the images are feed to the net in BGR format, NOT RGB
    channels = tf.unstack(img_standardized, axis=-1)
    img_standardized = tf.stack([channels[2], channels[1], channels[0]], axis=-1)

    return img_standardized
