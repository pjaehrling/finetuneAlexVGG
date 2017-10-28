import tensorflow.contrib.slim as slim

def load_weights(session, weights_path, restore_vars):
    """Load the initial weights from a checkpoint file
    Does not init the layers that we want to retrain

    Args:
        session: current tensorflow session
        weights_path:
        restore_vars:
    """
    print("Info: Restoring weights from checkpoint: {}".format(weights_path))
    # Create and call an operation that reads the network weights from the checkpoint file
    weight_init_op = slim.assign_from_checkpoint_fn(weights_path, restore_vars)
    weight_init_op(session)
    print("")