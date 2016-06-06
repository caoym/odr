# coding=utf-8
import random
import cv2
import numpy
from matplotlib.pyplot import *
from PIL import Image
from scipy import ndimage
import scipy
from skimage.filters import threshold_adaptive
from sklearn import preprocessing
from utils import AdjustPostion
import numpy.ma as ma

from scipy.ndimage import measurements, morphology
from utils.Words import separateWords, bounding_box
from skimage.morphology import skeletonize,medial_axis
from utils.thinning import zhangSuen
from skimage import segmentation
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from skimage import measure


__author__ = 'caoym'


def bounding_image(src):
    B = numpy.argwhere(src)
    (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1

    if (ystop-ystart)<16 or (xstop-xstart)<3:
        return None

    return src[ystart:ystop,xstart:xstop]


def getLines(src):

    #调整大小
    src = AdjustPostion.adjustSize(src)

    temp = src.copy()
    #调整水平
    src = cv2.Canny(src, 100, 200)
    src,slope = AdjustPostion.adjustSlope(src)

    #src = cv2.erode(src,cv2.getStructuringElement(cv2.MORPH_CROSS,(1, 3)) )
    #src = cv2.dilate(src,cv2.getStructuringElement(cv2.MORPH_CROSS,(1, 3)) )

    src = cv2.dilate(src,cv2.getStructuringElement(cv2.MORPH_RECT,(40, 3)) )
    src = cv2.erode(src,cv2.getStructuringElement(cv2.MORPH_RECT,(40, 3)) )

    src = cv2.erode(src,cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5)) )
    src = cv2.dilate(src,cv2.getStructuringElement(cv2.MORPH_RECT,(6, 5)) )

    src = 1*(src>128)

    labels_open, nbr_objects_open = measurements.label(src)

    #调整水平
    block_size = 40
    temp = threshold_adaptive(temp, block_size, offset=10)
    temp = numpy.array(temp,'uint8')*255
    temp = cv2.bitwise_not(temp)

    if slope != 0:
        temp = ndimage.rotate(temp, slope)
        #旋转后像素会平滑，重新二值化
        temp = cv2.bitwise_not(temp)
        temp = threshold_adaptive(temp, block_size,offset=20)
        temp = numpy.array(temp,'uint8')*255
        temp = cv2.bitwise_not(temp)

    lines = [];

    image  = numpy.zeros(numpy.array(src).shape)
    count = 0
    for i in range(1,nbr_objects_open+1):

        test = temp.copy()
        test[labels_open != i]=0
        box = bounding_box(test)
        x,y,w,h = box
        if h<10 or w<3:
            continue;
        #忽略靠近上下边的区域
        '''if y<2:
            continue
        if y+h> len(temp)-2:
            continue'''
        data = test[y:y+h, x:x+w]
        lines.append(data)

        copy = src.copy()*255.
        copy[labels_open != i]=0
        box = bounding_box(copy)
        x,y,w,h = box

        toerode = w/3
        if toerode <=1:
            continue

        copy = cv2.erode(copy,cv2.getStructuringElement(cv2.MORPH_RECT,(toerode, 1)) )
        copy = cv2.dilate(copy,cv2.getStructuringElement(cv2.MORPH_RECT,(toerode, 1)) )
        copy = 1*(copy>128)

        sub_labels_open, sub_nbr_objects_open = measurements.label(copy)
        if(sub_nbr_objects_open >1):
            for i in range(1,sub_nbr_objects_open+1):
                test = temp.copy()
                test[sub_labels_open != i]=0
                box = bounding_box(test)
                #count+=1
                #image[sub_labels_open == i] = count
                x,y,w,h = box
                if h<10 or w<3:
                    continue;
                #忽略靠近上下边的区域
                if y<2:
                    continue
                if y+h> len(temp)-2:
                    continue

                data = test[y:y+h, x:x+w]
                lines.append(data)

    '''figure()
    subplot(221)
    imshow(temp)
    subplot(222)
    imshow(image)
    subplot(223)
    imshow(labels_open)
    show()'''
    return lines

def eraseBlack(src):
    i = 0
    while True :
        h,w = src.shape
        if i >=w:
            break;

        if(numpy.max(src[:,i:i+1])<128):
            if i == w-1 or numpy.max(src[:,i+1:i+2])<128:
                src = numpy.delete(src,i,axis=1)
                continue
        i += 1

    return src


def getFeaturesFromImage(src, maxW=4.):
    '''
    :param src:
    :param maxW: 截取特征的最大宽度（以高度的倍数为单位，3倍差不多是3个字的宽度）
    :return:
    '''
    res = []

    lines = getLines(src)
    for lineData in lines:
        words = numpy.array(separateWords(lineData))
        averageH = numpy.average(words[:,3]);

        for i in range(0,len(words)):
            startX,startY,startW,startH = words[i]
            passed = 0
            pos = i
            clipTp = startY
            clipBt = startY+startH
            while (len == 0 or passed+startW<averageH*maxW) and pos<len(words):
                x,y,w,h = words[pos]
                clipLt = startX
                clipRt = x+w
                clipBt = max(y+h,clipBt)
                clipTp = min(y,clipTp)

                data = lineData[clipTp:clipBt,clipLt:clipRt]
                #
                #data2 = Image.fromarray(data).resize((64,32),Image.BILINEAR)

                #data = skeletonize(data/255)*255
                #data = zhangSuen(data/255)*255
                #删除连续的空白列
                data = eraseBlack(data)
                data = Image.fromarray(data).resize((96,32),Image.BILINEAR)
                #data = numpy.array(data)
                #data = skeletonize(data/255)*255
                #data = Image.fromarray(data)
                '''data = cv2.bitwise_not(data)
                data = threshold_adaptive(data, 40, offset=10)
                data = numpy.array(data,'uint8')*255
                data = cv2.bitwise_not(data)
                #data, distance = medial_axis(data, return_distance=True)
                data = skeletonize(data/255)*255'''

                #figure()
                #gray()
                #imshow(data2)
                #figure()
                #imshow(labels_open)
                #show()
                #objs = measurements.find_objects(labels_open,nbr_objects_open)

                '''figure()
                gray()
                imshow(data)
                show()'''

                #data = numpy.array(data,'uint8').tolist()
                #data = scaler.fit_transform(data)
                res.append(data);

                passed += w
                pos += 1
    '''figure()
    gray()
    for i in range(0,min(200,len(res))):
            if i>=20:
                break
            subplot(20,20,i+1)
            axis('off')
            img = res[i]
            img = numpy.array(img)
            imshow(img)
    show()'''
    return res;

def visualizationFeatures(src,maxW=4.):

    figure()
    gray()
    features = getFeaturesFromImage(src,maxW)

    i = 0
    for img in features:
        subplot(20,20,i+1)
        axis('off')
        imshow(img)
        i +=1
        if i == 400:
            figure()
            show();
            i = 0;

    show();
