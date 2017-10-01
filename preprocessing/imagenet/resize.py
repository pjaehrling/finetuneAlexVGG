import tensorflow as tf
from preprocessing import resize

MEAN = tf.constant([123.68, 116.78, 103.94], dtype=tf.float32) # IMAGENET

def preprocess_image(image, output_height, output_width, is_training=False):
    img_resized = resize.preprocess_image(image, output_height, output_width, is_training)
        
    # Subtract the imagenet mean (mean over all imagenet images)
    imgnet_mean = tf.reshape(MEAN, [1, 1, 3])
    img_cast = tf.cast(img_resized, dtype=tf.float32)
    img_standardized = tf.subtract(img_cast, imgnet_mean)

    return img_standardized
