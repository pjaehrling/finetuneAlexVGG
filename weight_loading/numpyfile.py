import numpy as np
import tensorflow as tf

def load_weights(session, weights_path, retrain_vars):
    """Load the initial weights from a numpy file
    Does not init the layers that we want to retrain

    Args:
        session: current tensorflow session
        weights_path:
        retrain_vars:
    """
    print("Info: Restoring weights from numpy file: {}".format(weights_path))
   
    # Load the weights into memory
    weights_dict = np.load(weights_path, encoding='bytes').item()
    # Loop over all layer ops
    for op_name in weights_dict:
        op_name_string = op_name if isinstance(op_name, str) else op_name.decode('utf8')

        # Check if the layer is one of the layers that should be reinitialized
        if op_name_string not in retrain_vars:
            print("  restore: {}".format(op_name_string))
            with tf.variable_scope(op_name_string, reuse=True):
                # Loop over list of weights/biases and assign them to their corresponding tf variable
                for data in weights_dict[op_name]:
                    # Biases
                    if len(data.shape) == 1:
                        var = tf.get_variable('biases', trainable = False)
                        session.run(var.assign(data)) 
                    # Weights
                    else:
                        var = tf.get_variable('weights', trainable = False)
                        session.run(var.assign(data))
        else:
            print("  skip: {}".format(op_name_string))
    
    print("")