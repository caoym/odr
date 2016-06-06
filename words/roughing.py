# coding=utf-8
import os
from utils.Words import separateWords

__author__ = 'caoym'
import random
import matplotlib.patches as patches
import scipy

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
from utils.AdjustPostion import classify, getLines
import matplotlib.pyplot as plt
from utils.rof import denoise
from skimage import morphology


def uniteRectsOnce(pos, contours):
    count = len(contours)
    if(pos>count-2):
        return contours,False
    x,y,w,h = contours[pos]
    i = pos+1
    while i <len(contours):
        x1,y1,w1,h1 = contours[i]
        #相交
        _,intersert = intersectRect((x,y,w,h),contours[i])
        if intersert:
            contours[pos] = joinRect((x,y,w,h),(x1,y1,w1,h1))
            contours = numpy.delete(contours,i,axis=0)
            return contours,True
        if x1>x+w:
                return contours,False
        i+=1
    return contours,False

def joinRect(r0,r1):
    x,y,w,h = r0
    x1,y1,w1,h1=r1
    l = min(x,x1)
    t = min(y,y1)
    r = max(x1+w1,x+w)
    b = max(y1+h1,y+h)
    return [l,t,r-l,b-t]

def intersectRect(r0,r1):
    x,y,w,h = r0
    x1,y1,w1,h1=r1

    l = max(x,x1)
    r = min(x+w, x1+w1)

    t = max(y,y1)
    b = min(y+h,y1+h1)
    if l>r or t>b:
        return [0,0,0,0],False
    else:
        return [l,t,t-l,b-t],True


#获取最可能是文字的区域，用于提取训练样本
def getMostLikelyWords(src):
    mask = src.copy()
    mask = cv2.Canny(mask, 100, 200)
    #mask = mask + src
    contours,_  = cv2.findContours(mask.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    rects=[]
    for i in contours:
        rc = cv2.boundingRect(i)
        if rc[3]>50 or rc[2]>50:
            continue
        rects+=[rc]
    contours = numpy.array(rects)
    #return contours
    #按x排序
    contours = contours[numpy.argsort(contours[:,0])]

    #合并相交的矩形
    pos = 0
    while pos < len(contours):
        contours,did = uniteRectsOnce(pos,contours)
        if not did:
            pos += 1

    avgW = numpy.average(contours[:,2])
    avgH = numpy.average(contours[:,3])

    #找出最可能是字的矩形
    candidates = []
    for i in contours:
        x,y,w,h = i

        if w < avgW or h < avgH:
            continue
        if w<h/1.5 or w>h*1.5:
            continue
        candidates += [[w, h]]
    candidates = numpy.array(candidates)

    (n, bins) = numpy.histogram(candidates[:,1], bins=20, normed=True)

    #最普遍的高度和宽度
    for i in range(0,20):
        max = n.argmax(axis=0)
        mustH = bins[max]
        if mustH > avgH and mustH>10:
            break
        else:
           n =  numpy.delete(n,max,axis=0)
           bins =  numpy.delete(bins,max,axis=0)

    (n, bins) = numpy.histogram(candidates[:,0], bins=20, normed=True)
    for i in range(0,20):
        max = n.argmax(axis=0)
        mustW = bins[max]
        if mustW > avgW and mustW>10:
            break
        else:
           n =  numpy.delete(n,max,axis=0)
           bins =  numpy.delete(bins,max,axis=0)
    #pylab.hist(candidates, bins=10)
    #mustW = bins[n.argmax(axis=0)]
    #return  contours

    newcontours = []
    #return contours
    for i in contours:
        x,y,w,h = i
        if w > mustW and w>h*1.5:
            rects = separateWords(src[y:y+h,x:x+w])
            for r in rects:
                x1,y1,w1,h1 = r
                if h1 > mustH*1.5 or h1 < mustH*0.7: #过高或过低
                    continue
                if w1 > mustW*1.5 or w1 < mustW*0.7: #过宽或过窄
                    continue
                r1 = float(w1)/float(h1)
                if r1 > mustH/mustW*1.5 or r1 < mustH/mustW*0.7: #长宽比不对
                    continue
                newcontours += [[x+x1,y+y1,w1,h1]]
            continue
        if h > mustH*1.5 or h < mustH*0.7: #过高或过低
            continue
        elif w > mustW*1.5 or w < mustW*0.7: #过宽或过窄
            continue
        r = float(w)/float(h)
        if r > mustH/mustW*1.5 or r < mustH/mustW*0.7: #长宽比不对
            continue
        newcontours += [i]

    return numpy.array(newcontours)

def getMostLikelyWordsWithFiles(fr):

    from_dir = os.listdir(fr)

    for f in from_dir:
        from_filr = fr+'/'+f
        if not os.path.isfile(from_filr):
            continue
        print("processing %s..."%from_filr)
        src = Image.open(from_filr).convert('L')
        src = numpy.array(src,'uint8')

        src = cv2.bitwise_not(src)
        #retval, src = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        src = AdjustPostion.adjustSize(src)

        words = getMostLikelyWords(src)
        to_dir = fr+'/'+f+'_words'
        if not os.path.isdir(to_dir):
            os.makedirs(to_dir)
        pos = 0
        gray()
        for word in words:
            x,y,w,h = word
            to_filr = "%s/%d.png"%(to_dir,pos)
            imsave(to_filr, src[y:y+h,x:x+w],)
            pos +=1

