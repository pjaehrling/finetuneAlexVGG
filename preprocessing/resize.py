import tensorflow as tf

def preprocess_image(image, output_height, output_width, is_training=False):
    # Resize
    input_size = tf.constant([output_width, output_height], dtype=tf.int32)
    img_resized = tf.image.resize_images(image, input_size)

    return img_resized
