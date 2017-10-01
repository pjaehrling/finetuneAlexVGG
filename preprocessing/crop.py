import tensorflow as tf

def preprocess_image(image, output_height, output_width, is_training=False):
    # Crop
    return tf.image.resize_image_with_crop_or_pad(image, output_width, output_height)
