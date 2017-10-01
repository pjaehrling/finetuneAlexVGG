import tensorflow as tf
from preprocessing import crop

MEAN = tf.constant([123.68, 116.78, 103.94], dtype=tf.float32) # IMAGENET

def preprocess_image(image, output_height, output_width, is_training=False):
    # Crop
    img_crop = crop.preprocess_image(image, output_height, output_width, is_training)
        
    # Subtract the imagenet mean (mean over all imagenet images)
    imgnet_mean = tf.reshape(MEAN, [1, 1, 3])
    img_cast = tf.cast(img_crop, dtype=tf.float32)
    img_standardized = tf.subtract(img_cast, imgnet_mean)

    return img_standardized
