save-deep
=========

This is a package for saving activations of a specified set of layers for a given Caffe-format deep neural network model, in response to each image in a specified directory. This is useful for doing experiments related to how neurons in specific layers respond to particular stimuli. This saves output to a pandas DataFrame, where every row has a corresponding 'image', as well as tensors indexed by blog name (e.g. 'fc6', 'fc7', ...).

Since PyCaffe can be a bit of a hassel to setup, everything can be run from within a Docker image. In other words, all you need to install is Docker, then just run the `run_docker.sh` script! This uses a [pre-packaged Docker image](https://github.com/saiprashanths/dl-docker) with Caffe and other needed libraries pre-installed.

The primary script to run is `save.py`, which takes 4 command-line args:
* `-m`/`--model`: Model directory.
* `-b`/`--blobs`: List of blobs to save, e.g. 'fc6,fc7,fc7', or 'all'.
* `-i`/`--imgs`: Load input images from this directory.
* `-o`/`--out`: Output data to this file (.pk extension).

For the '--model' argument to `save.py`, you'll need 3 files in the directory: `deploy.prototxt`, `weights.caffemodel`, and `mean.npy`. `.prototxt` and `.caffemodel` are the standard distribution format for deep neural models trained using Caffe. `mean.npy` is a vector of mean BGR values over all images in the training set, so this is a 3-index Numpy array.



Requires
--------
* [Docker](https://www.docker.com/products/docker)
* ~3gb of space

OR

* [Pandas](http://pandas.pydata.org/)
* [PyCaffe](http://installing-caffe-the-right-way.wikidot.com/start)


Usage Instructions
------------------


### Download an example convolutional neural network (CNN) model to use

#### Option 1: AlexNet through caffe

If you don't already have caffe downloaded, clone the git repo:

    $ git clone https://github.com/BVLC/caffe

The following code downloads [AlexNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet) to your caffe folder, then creates the appropriate symbolic links for each of the three needed files:

    $ CAFFE_ROOT="/path/to/caffe"
    $ $CAFFE_ROOT/scripts/download_model_binary.py $CAFFE_ROOT/models/bvlc_reference_caffenet
    $ mkdir -p models/alexnet
    $ ln -s $CAFFE_ROOT/models/bvlc_reference_caffenet/deploy.prototxt ./models/alexnet/deploy.prototxt
    $ ln -s $CAFFE_ROOT/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel ./models/alexnet/weights.caffemodel
    $ ln -s $CAFFE_ROOT/python/caffe/imagenet/ilsvrc_2012_mean.npy ./models/alexnet/mean.npy
    $ mkdir out
    
#### Option 2: Other caffe models
    
Alternatively, download a caffe model from the [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo). Make sure to include `deploy.prototxt`, `weights.caffemodel`, and `mean.npy` files in your model directory.

Note that many models will not include `mean.npy` -- you will have to create one, for example if the BGR (blue, green, red) mean is `[60, 80, 100]`:

    $ python
    $ >>> import numpy as np
    $ >>> mean = np.array([60,80,100])
    $ >>> np.save('./model/mean', mean)

### Set up images folder

All you need for this is a folder with your images in it.

We might also write symbolic links in the save-deep `directory` to our images folder as well, for convenience:

    $ ln -s /path/to/images ./images
    
### Create directory for outputting activation data

    $ mkdir ./out

    
### Run the model

#### Option 1: Run through docker

For less savvy users, all you have to do after setting up your `model`, `images`, and `output` directories is run the `run_docker.sh` script with your parameters. For example:

    $ sh run_docker.sh /Users/eric/code/save-deep/docker1 /Users/eric/code/save-deep/models/alexnet /Users/eric/code/save-deep/images
    
**Note**: Make sure you use the *full file path* for your directories!


#### Option 2: Install PyCaffe and run python script directly

You first need to [install and setup pycaffe](http://installing-caffe-the-right-way.wikidot.com/start) on your own machine, and install the pandas python package. You may then run the `save.py` script directly, saving just the fully connected layers of AlexNet by executing:

    $ python save.py -m ./model -b fc6,fc7,fc7 -i ./images -o ./out





Notes
-----
* The first 2 layers take up a lot of space, the following convolution layers quite a bit of space, and the fully connected layers, very little space. The amount of space on disk taken up by activation data for a single image, for each layer of AlexNet, is approximately:

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

Space is linear with the number of neurons, at approximately 3 bytes per image for each neuron.

