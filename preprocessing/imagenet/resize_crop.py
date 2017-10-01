import tensorflow as tf
from preprocessing import resize_crop

MEAN = tf.constant([123.68, 116.78, 103.94], dtype=tf.float32) # IMAGENET

def preprocess_image(image, output_height, output_width, is_training=False):
    img_resize_crop = resize_crop.preprocess_image(image, output_height, output_width, is_training)
        
    # Subtract the imagenet mean (mean over all imagenet images)
    imgnet_mean = tf.reshape(MEAN, [1, 1, 3])
    img_float = tf.to_float(img_resize_crop)
    img_standardized = tf.subtract(img_float, imgnet_mean)

    return img_standardized
