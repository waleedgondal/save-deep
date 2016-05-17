save-deep
=========

This is a package for saving activations of a specified set of layers for a given Caffe-format deep neural network model, in response to each image in a specified directory. This is useful for doing experiments related to how neurons in specific layers respond to particular stimuli. This saves output to a pandas DataFrame, where every row has a corresponding 'image', as well as tensors indexed by blog name (e.g. 'fc6', 'fc7', ...).

The primary script to run is `save.py`, which takes 4 command-line args:
* `-m`/`--model`: Model directory.
* `-b`/`--blobs`: List of blobs to save, e.g. 'fc6,fc7,fc7'.
* `-i`/`--imgs`: Load input images from this directory.
* `-o`/`--out`: Output data to this file (.pk extension).

For the '--model' argument to `save.py`, you'll need 3 files in the directory: `deploy.prototxt`, `weights.caffemodel`, and `mean.npy`. `.prototxt` and `.caffemodel` are the standard distribution format for deep neural models trained using Caffe. `mean.npy` is a vector of mean BGR values over all images in the training set, so this is a 3-index Numpy array.

The following code downloads AlexNet to your caffe folder, then creates the appropriate symbolic links for each of the three needed files:

    $ CAFFE_ROOT="/path/to/caffe"
    $ $CAFFE_ROOT/scripts/download_model_binary.py $CAFFE_ROOT/models/bvlc_reference_caffenet
    $ mkdir -p models/alexnet
    $ ln -s $CAFFE_ROOT/models/bvlc_reference_caffenet/deploy.prototxt models/alexnet/deploy.prototxt
    $ ln -s $CAFFE_ROOT/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel models/alexnet/weights.caffemodel
    $ ln -s $CAFFE_ROOT/python/caffe/imagenet/ilsvrc_2012_mean.npy models/alexnet/mean.npy
    $ mkdir out


Requires
--------
* Pandas
* PyCaffe


Notes
-----
* I'm working on wrapping this in a docker image, so less savvy users won't have to deal with installing PyCaffe locally on their machines.
* The first 2 layers take up a lot of space, the following convolution layers quite a bit of space, and the fully connected layers, very little space. The amount of space taken up by activation data by each layer of AlexNet, for each image, is approximately:

| Depth | Layer     | Shape         | Mem/Img |
|-------|-----------|---------------|---------|
| 1.    | **conv1** | (96, 55, 55)  | 894 kB  |
| 2.    | **pool1** | (96, 27, 27)  | 215 kB  |
| 3.    | **norm1** | (96, 27, 27)  | 215 kB  |
| 4.    | **conv2** | (256, 27, 27) | 574 kB  |
| 5.    | **pool2** | (256, 13, 13) | 133 kB  |
| 6.    | **norm2** | (256, 13, 13) | 133 kB  |
| 7.    | **conv3** | (384, 13, 13) | 200 kB  |
| 8.    | **conv4** | (384, 13, 13) | 200 kB  |
| 9.    | **conv5** | (256, 13, 13) | 133 kB  |
| 10.   | **pool5** | (256, 6, 6)   | 28 kB   |
| 11.   | **fc6**   | (4096,)       | 12 kB   |
| 12.   | **fc7**   | (4096,)       | 12 kB   |
| 13.   | **fc8**   | (1000,)       | 3 kB    |
| 14.   | **prob**  | (1000,)       | 3 kB    |


conv1 (96, 55, 55)
pool1 (96, 27, 27)
norm1 (96, 27, 27)
conv2 (256, 27, 27)
pool2 (256, 13, 13)
norm2 (256, 13, 13)
conv3 (384, 13, 13)
conv4 (384, 13, 13)
conv5 (256, 13, 13)
pool5 (256, 6, 6)
fc6   (4096,)
fc7   (4096,)
fc8   (1000,)
prob  (1000,)
