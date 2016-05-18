import caffe
import numpy as np
import pandas as pd
import pickle
import os
from optparse import OptionParser


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-m", "--model", dest="model_root", 
                      type="string", default="models/bvlc_reference_caffenet",
                      help="Model directory.")
    parser.add_option("-b", "--blobs", dest="save_blobs", 
                      type="string", default="all",
                      help="List of blobs to save, e.g. 'fc6,fc7,fc7'.")
    parser.add_option("-i", "--imgs", dest="img_dir", 
                      type="string", default="images",
                      help="Load input images from this directory")
    parser.add_option("-o", "--out", dest="out_file", 
                      type="string", default="out/alexnet.pk",
                      help="Output data to this file (.pk extension).")

    (options, args) = parser.parse_args()
    
    caffe.set_mode_cpu()

    model_root = options.model_root
    save_blobs = options.save_blobs.split(',')
    img_dir = './images/'

    out_file   = options.out_file
  

    # Load caffe model
    model_def     = model_root + '/deploy.prototxt'
    model_weights = model_root + '/weights.caffemodel'

    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)

    # Load RGB means for subtraction
    mu = np.load(model_root + '/mean.npy')


    # Create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))     # rearrange dimensions
    transformer.set_mean('data', mu)               # subtract mean dataset BGR values
    transformer.set_raw_scale('data', 255)         # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap from RGB to BGR


    # Extract blobs for ever image in specified directory
    img_files = os.listdir(img_dir)

    batch_size = net.blobs['data'].data.shape[0]
    n_batches  = int(round(len(img_files) / float(batch_size)))

    rows = []

    for b in range(n_batches):
        images = []
        batch_files = img_files[b:b+batch_size]

        # Load & transform image data for this batch
        for img_file in batch_files:
            image = caffe.io.load_image(img_dir + '/' + img_file)
            transformed_image = transformer.preprocess('data', image)
            images.append(transformed_image)

        # Add dummy images if we have fewer than `batch_size` 
        #  images to process in this batch
        diff = batch_size - len(images) 
        if diff > 0:
            images += [images[-1]] * diff
            batch_files += ['']

        # Input batch data to CNN, forward propogate information
        net.blobs['data'].data[...] = np.stack(images, axis=0)
        net.forward()

        # One row per image
        for i, img_file in enumerate(batch_files):
            if save_blobs[0] == 'all':
                row = {blob:net.blobs[blob].data[i] for blob in net.blobs if blob != 'data'}
            else:
                row = {blob:net.blobs[blob].data[i] for blob in save_blobs}
            row['image'] = img_file
            rows.append(row)

    # Save blobs
    df = pd.DataFrame(rows)
    df.to_pickle(out_file)
