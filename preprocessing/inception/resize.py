import tensorflow as tf
from preprocessing import resize

def preprocess_image(image, output_height, output_width, is_training=False):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = resize.preprocess_image(image, output_height, output_width, is_training)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image
