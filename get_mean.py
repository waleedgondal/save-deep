import cv2
"""

TODO: instructions . . .

"""
import numpy as np
from optparse import OptionParser

if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-i", "--in", dest="img_dir",
                      type="string", default="./images",
                      help="Images directory.")
    parser.add_option("-o", "--out", dest="out_dir",
                      type="string", default="./model",
                      help="Out directory.")
    (options, args) = parser.parse_args()
    img_dir = options.img_dir
    img_files = os.listdir(img_dir)
    N = len(img_files)

    Mu = np.zeros(3)
    for img_file in img_files:
        img = cv2.imread(img_dir + '/' + img_file).astype(float)
        Mu += np.mean(img, axis=(0,1)) / N

    np.save(options.out_dir + 'mean.npy', Mu)

