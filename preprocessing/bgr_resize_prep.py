import tensorflow as tf
from tensorflow.python.ops import math_ops

MEAN = tf.constant([123.68, 116.78, 103.94], dtype=tf.float32) # IMAGENET

def preprocess_image(image, output_height, output_width, is_training=False):
    # Crop
    input_size = tf.constant([output_width, output_height], dtype=tf.int32)
    img_resized = tf.image.resize_images(image, input_size)
        
    # Subtract the imagenet mean (mean over all imagenet images)
    imgnet_mean = tf.reshape(MEAN, [1, 1, 3])
    img_cast = math_ops.cast(img_resized, dtype=tf.float32)
    img_standardized = math_ops.subtract(img_cast, imgnet_mean)

    # e.g. in my alexnet implementation the images are feed to the net in BGR format, NOT RGB
    channels = tf.unstack(img_standardized, axis=-1)
    img_standardized = tf.stack([channels[2], channels[1], channels[0]], axis=-1)

    return img_standardized
