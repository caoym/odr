# coding=utf-8
from PIL import Image
import cv2
import math
from matplotlib.pyplot import *
import numpy
import pylab
from scipy import ndimage
from scipy.cluster.vq import kmeans,whiten,vq
from scipy.ndimage import filters
from sympy import Point, Line
from scipy import signal
from utils import AdjustPostion
from utils.AdjustPostion import getLines
import matplotlib.pyplot as plt
from utils.Words import getWords, separateWords
from utils.rof import denoise
from words.classify import classify
from words.roughing import getMostLikelyWords, getMostLikelyWordsWithFiles

__author__ = 'caoym'

def threshold(src):
    return  src
    (n, bins) = numpy.histogram(src.flatten(), bins=256)

    figure()

    pylab.hist(src.flatten(), bins=10)
    n = numpy.column_stack((numpy.arange(0,len(n)), n))

    features = whiten(n)
    center, f = kmeans(features, 2)

    figure()
    plot(features[:,0],features[:,1])
    t =  (center[0][0]+center[1][0])/2
    retval, src = cv2.threshold(src, t, 255, cv2.THRESH_BINARY)
    return src
def test():

    src = Image.open('D:\\cloud\\2222.jpg').convert('L')
    src = numpy.array(src,'uint8')

    src = cv2.bitwise_not(src)
    src = AdjustPostion.adjustSize(src)


    figure()
    gray()
    imshow(src)
    show()
    #颜色
    #src = threshold(src)
    #show()
    #return
    #retval, src = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    words = getMostLikelyWords(src)

    figure()
    gray()

    imshow(src)
    pos = 0
    for word in words:
        x,y,w,h = word
        plot([x,x+w,x+w,x,x], [y,y,y+h,y+h,y], '-')
        #ax.imshow(src[y:y+h,x:x+w])
        pos += 1
    show()

def test2():
    src = Image.open('D:\\cloud\\2222.jpg').convert('L')
    src = numpy.array(src,'uint8')
    src = cv2.bitwise_not(src)
    words = separateWords(src)
    figure()
    gray()

    imshow(src)
    pos = 0
    for word in words:
        x,y,w,h = word
        plot([x,x+w,x+w,x,x], [y,y,y+h,y+h,y], '-')
        pos += 1

    new = [];
    for word in words:
        x,y,w,h = word
        img = src[y:y+h,x:x+w];
        retval, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        img = Image.fromarray(img).resize((32,32))
        new.append(numpy.array(img).flatten());
    new = numpy.array(new);
    figure()
    imshow(new)
    show()
    imsave("d:\\a.png",new)


if __name__ == '__main__':
    #classify("D:\\data\\words\\0ff41bd5ad6eddc41bcb0a1e3bdbb6fd52663322.jpg_words")
    test2()
    #getMostLikelyWordsWithFiles("D:/data/words")