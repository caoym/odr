# coding=utf-8
import os
import cv2
from matplotlib.pyplot import imsave, gray, figure,imshow,show
import numpy
from scipy.cluster.vq import kmeans,whiten,vq
import shutil
__author__ = 'caoym'
from PIL import Image

def classify(fr):
    from_dir = os.listdir(fr)
    features = []
    gray()
    files = []
    for f in from_dir:
        from_file = fr+'/'+f
        if os.path.isfile(from_file):
            src = Image.open(from_file).convert('L')
            src = src.resize((32,32))
            src = numpy.array(src,'uint8')
            src = cv2.bitwise_not(src)
            retval, src = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

            src = src.flatten()
            features.append(src)
            files.append(from_file)


    features = numpy.vstack(features)
    features = whiten(features)
    center, f = kmeans(features, 2000)
    code, dis = vq(features, center)
    to_dir = fr+'/classify'
    if not os.path.isdir(to_dir):
        os.makedirs(to_dir)


    for i in range(0,len(code)):
        type = code[i]
        to_file = "%s/%d_%d.png"%(to_dir,type,i)
        shutil.copy(files[i],to_file)




