# Finetune AlexNet & VGG with Tensorflow

AlexNet and VGG16 model implementations for Tensorflow, with a validation and finetune/retrain script.

## Requirements

- Python 2.7
- TensorFlow >= 1.13rc0 (I guess everything from version 1.0 on will work)
- Numpy

## Content

- `validate.py`: Script to validate the implemented models and the downloaded weights 
- `finetune.py`: Script to run the finetuning process
- `helper/*`: Contains helper scripts/classes to load images and run the retraining
- `models/*`: Contains a parent model class and different model implementations (AlexNet, VGG)
- `images/*`: contains 4 example images, used in the validation script

# Weights:
- AlexNet: http://www.cs.toronto.edu/%7Eguerzhoy/tf_alexnet/bvlc_alexnet.npy
- VGG: https://mega.nz/a96db891-9e0d-1644-bee9-b2679aa26378

## Usage

### Validate the model implementations and weights
```
python validate.py -model alex
...
python validate.py -model vgg
```

### Run finetuning/retraining on selected layers
```
python finetune.py -image_path /path/to/images -model alex
...
python finetune.py -image_path /path/to/images -model vgg
```

`/path/to/image` should point to a folder with a set of sub-folders, each named after one of your final categories and containing only images from that category.

Another way is to provide a file with a list of image-paths and labels using the `-image_file` argument instead of `-image_path`. 
e.g.
```
cat /path/to/cat1.jpg
cat /path/to/cat2.jpg
dog /path/to/dog1.jpg
...
```

With every epoch a checkpoint file can be written to save the training progress.
It's possible to start the training from a saved checkpoint file by providing a path to that file when calling the finetune script.
```
python finetune.py -image_path /path/to/images -model alex
...
python finetune.py -image_path /path/to/images -model vgg -ckpt /path/to/file.ckpt
```

TensorFlows summaries are implemented so that TensorBoard can be used.

# Useful sources:
- https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html
- https://kratzert.github.io/2017/06/15/example-of-tensorflows-new-input-pipeline.html
- https://www.cs.toronto.edu/~frossard/post/vgg16/
- https://github.com/machrisaa/tensorflow-vgg
