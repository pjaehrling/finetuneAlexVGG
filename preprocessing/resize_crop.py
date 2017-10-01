import tensorflow as tf

def preprocess_image(image, output_height, output_width, is_training=False):
    # Get the input Dimensions
    input_shape = tf.shape(image)
    input_height = tf.to_float(input_shape[0])
    input_width = tf.to_float(input_shape[1])

    # Find out which side has the smallest scalling factor, so we resize by this
    scale_height = tf.to_float(output_height) / input_height
    scale_width  = tf.to_float(output_width) / input_width
    scale = tf.cond(tf.greater(scale_height, scale_width), lambda: scale_height,lambda: scale_width)

    new_height = tf.to_int32(input_height * scale)
    new_width = tf.to_int32(input_width * scale)

    # Resize (keep ratio) and Crop to fit output dimensions
    img_resize = tf.image.resize_images(image, [new_height, new_width])
    img_resize_crop = tf.image.resize_image_with_crop_or_pad(img_resize, output_height, output_width)

    return img_resize_crop
