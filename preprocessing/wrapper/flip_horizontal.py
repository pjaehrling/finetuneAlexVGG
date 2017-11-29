import tensorflow as tf

def wrap(image_prep):
    def preprocess_image(image, output_height, output_width, is_training=False):
        image = image_prep.preprocess_image(image, output_height, output_width, is_training)
        image = tf.image.flip_left_right(image)
        return image

    result = lambda: None # dirty hack
    result.preprocess_image = preprocess_image
    return result