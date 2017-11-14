#
# Author: Philipp Jaehrling
#
import os
import argparse
import helper.data_provider as data_provider

from helper.feature.trainer import FeatureTrainer

# Input params
VALIDATION_RATIO = 5 # e.g. 5 -> every 5th element = 1/5 = 0.2 = 20%
USE_SUBFOLDER = True
SKIP_FOLDER = ['yiwen']

# Learning params
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
BATCH_SIZE = 32

# Network params
KEEP_PROB = 0.5 # [0.5]
CHECKPOINT_DIR = '../checkpoints/features'

# HARDWARE USAGE
DEVICE = '/cpu:0'
MEMORY_USAGE = 1.0

def train(data, layer_inputs, ckpt_dir, write_checkpoint_on_each_epoch, init_from_ckpt, use_adam_optimizer):
    """
    Args:
        data:
        layer_input:
        ckpt_dir:
        write_checkpoint_on_each_epoch:
        init_from_ckpt:
        use_adam_optimizer:
    """
    # TODO: Get the input shape by looking at one file
    # f = open('foo.txt', 'r')
    # f.readlines()

    trainer = FeatureTrainer(data, write_checkpoint_on_each_epoch)
    trainer.run(
        layer_inputs,
        NUM_EPOCHS,
        LEARNING_RATE,
        BATCH_SIZE,
        KEEP_PROB,
        MEMORY_USAGE,
        DEVICE,
        ckpt_dir,
        init_from_ckpt,
        use_adam_optimizer
    )

def main():
    """
    Main
    """
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-layer_inputs',
        type=int,
        nargs='+',
        required=True,
        help='The dimensions of the FullyConnected layers.\n' +
             '"-layer_inputs 4049" will create one FC layer: : [4096] -> FC1 -> [classes].\n' +
             '"-layer_inputs 4049 1024" will create two FC layers: [4096] -> FC1 -> [1024] -> FC2 -> [classes].\n'
    )
    parser.add_argument(
        '-feature_dir',
        type=str,
        default='',
        help='Folder with trainings/validation features'
    )
    parser.add_argument(
        '-feature_file',
        type=str,
        default='',
        help='File with a list of trainings/validation features and their labels'
    )
    parser.add_argument(
        '-write_checkpoint_on_each_epoch',
        default=False,
        help='Write a checkpoint file on each epoch (default is just once at the end',
        action='store_true' # whenever this option is set, the arg is set to true
    )
    parser.add_argument(
        '-init_from_ckpt',
        type=str,
        default='',
        help='Load this checkpoint file to continue training from this point on'
    )
    parser.add_argument(
        '-use_adam_optimizer',
        default=False,
        help='Use Adam optimizer instead of GradientDecent',
        action='store_true' # whenever this option is set, the arg is set to true
    )
    
    args = parser.parse_args()
    layer_inputs = args.layer_inputs
    feature_dir = args.feature_dir
    feature_file = args.feature_file
    write_checkpoint_on_each_epoch = args.write_checkpoint_on_each_epoch
    init_from_ckpt = args.init_from_ckpt
    use_adam_optimizer = args.use_adam_optimizer

    # Load images
    if not feature_dir and not feature_file:
        print('Provide one of the following options to load images \'-feature_file\' or \'-feature_dir\'')
        return None
    elif feature_dir: 
        if not os.path.exists(feature_dir):
            print('Root directory \'%s\' not found' %feature_dir)
            return None
        else:
            data = data_provider.load_features_by_subfolder(feature_dir, VALIDATION_RATIO, SKIP_FOLDER, use_subfolder=USE_SUBFOLDER)
    else:
        if not os.path.exists(feature_file):
            print('Feature list file \'%s\' not found' %feature_file)
            return None
        else:
            data = data_provider.load_by_file(feature_file, VALIDATION_RATIO)

    # Make sure we have enough images to fill at least one training/validation batch
    if data['training_count'] < BATCH_SIZE:
        print('Not enough training features')
        return None

    if data['validation_count'] < BATCH_SIZE:
        print('Not enough validation features')
        return None

    # Make sure the checkpoint dir exists
    ckpt_dir = CHECKPOINT_DIR
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # Start retraining/finetuning
    train(data, layer_inputs, ckpt_dir, write_checkpoint_on_each_epoch, init_from_ckpt, use_adam_optimizer)

if __name__ == '__main__':
    main()