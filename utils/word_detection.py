# coding=utf-8
from PIL import Image
import cv2
import numpy
import random
from utils import AdjustPostion

__author__ = 'caoym'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import data
from skimage.filters import threshold_otsu, threshold_adaptive
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing, square
from skimage.measure import regionprops
from skimage.color import label2rgb

from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage import data, img_as_float

from skimage.morphology import watershed

from skimage.filters import sobel
from skimage.segmentation import slic, join_segmentations
from skimage.morphology import watershed
from skimage.color import label2rgb
from skimage import data, img_as_float
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt
from skimage.color import rgb2gray

from skimage import data
from skimage.morphology import disk
from skimage.filters import rank

import matplotlib.pyplot as plt

def find_words(image):
    image = numpy.array(image,'uint8')

    image = AdjustPostion.adjustSize(image)

    block_size = 40
    image = threshold_adaptive(image, block_size, offset=10)
    image = numpy.array(image,'uint8')*255
    im = cv2.bitwise_not(image)
    #image = closing(image, square(5))

    # image_max is the dilation of im with a 20*20 structuring element
    # It is used within peak_local_max function
    image_max = ndi.maximum_filter(im, size=20, mode='constant')

    # Comparison between image_max and im to find the coordinates of local maxima
    coordinates = peak_local_max(im, min_distance=20)

    # display results
    fig, ax = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True,
                           subplot_kw={'adjustable': 'box-forced'})
    ax1, ax2, ax3 = ax.ravel()
    ax1.imshow(im, cmap=plt.cm.gray)
    ax1.axis('off')
    ax1.set_title('Original')

    ax2.imshow(image_max, cmap=plt.cm.gray)
    ax2.axis('off')
    ax2.set_title('Maximum filter')

    ax3.imshow(im, cmap=plt.cm.gray)
    ax3.autoscale(False)
    ax3.plot(coordinates[:, 1], coordinates[:, 0], 'r.')
    ax3.axis('off')
    ax3.set_title('Peak local max')

    fig.tight_layout()

    plt.show()
    return


if __name__ == '__main__':
    src = Image.open(u"D:\\data\\samples\\物理-力学-质点的直线运动\\1454081683.jpg").convert('L')
    find_words(src)