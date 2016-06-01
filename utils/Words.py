# coding=utf-8
import random
import matplotlib.patches as patches
import scipy

__author__ = 'caoym'

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

from skimage import data
from skimage.morphology import disk
from skimage.filters import rank
from skimage.morphology import closing, square

__author__ = 'caoym'

def bounding_box(src):
    B = numpy.argwhere(src)
    if B.size == 0:
        return [0,0,0,0]
    (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1

    return [xstart,ystart,xstop-xstart,ystop-ystart]

def getDistance2(p1,p2 ,p0): #p0 is the point
    x1,y1 = p1
    x2,y2 = p2
    x3,y3 = p0
    return abs(y3-y1)+abs(y3-y2)

def getDistance(p1,p2 ,p0): #p0 is the point
    x1,y1 = p1
    x2,y2 = p2
    x3,y3 = p0
    px = x2-x1
    py = y2-y1

    something = px*px + py*py
    if something==0:
        return 0
    u =  ((x3 - x1) * px + (y3 - y1) * py) / float(something)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = x1 + u * px
    y = y1 + u * py

    dx = x - x3
    dy = y - y3

    # Note: If the actual distance does not matter,
    # if you only want to compare what this function
    # returns to other results of this function, you
    # can just return the squared distance instead
    # (i.e. remove the sqrt) to gain a little performance

    dist = math.sqrt(dx*dx + dy*dy)
    #if y1<=1 or y2<=1:
    #    dist+=2
    return dist

#如果下一个更低，删除下一个
def delLowOnce(points,id,threshold):
    if id<len(points)-1 and id>0:
        dist = getDistance((points[id-1][0],points[id-1][1]),(points[id+1][0],points[id+1][1]),(points[id][0],points[id][1]))
        if dist<threshold:
            res,d = delLowOnce(points,id+1,dist)
            if not d:
                res = numpy.delete(points,id,axis=0)
            return res,True
    return points,False

def delLow(points,threshold):
    id = 1
    while id<len(points)-1:
        points,_ = delLowOnce(points,id, threshold)
        id +=1
    return points

def getSeparaters2(src):
    #src = src**2
    #plot(numpy.arange(src.shape[:2][0]),src,'r-')
    src = signal.detrend(src)
    #plot(numpy.arange(src.shape[:2][0]),src,'g-')
    #找出拐点
    trend = False #False 下降，True上涨
    preval = 0 #上一个值
    points = [] #拐点
    pos = 0
    for i in src:
        if preval != i:
            if trend != (i>preval):
                trend = (i>preval)
                points += [[pos if pos == 0 else pos-1,preval]]
        pos = pos+1
        preval = i
    if len(points) <4:
        return [[0,len(src)]]
    points += [[pos,0]]
    #计算每个点的振幅(与前后两个拐点直线的垂直距离)
    pending = []
    count = len(points)
    for i in range(0, count):
        if i ==0 or i == count-1:
            continue
        dist = getDistance((points[i-1][0], points[i-1][1]), (points[i+1][0], points[i+1][1]),(points[i][0], points[i][1]))
        pending += [[i,dist, points[i][0],points[i][1]]]
    points = numpy.array(points)
    pending = numpy.array(pending)
    #plot(points[:,0],points[:,1],'b*')

    #plot(pending[:,2],pending[:,1],'r-')
    average = numpy.average(pending[:,1])
    max = numpy.max(pending[:,1])
    min = numpy.min(pending[:,1])
    threshold = average*0.5
    #todels = numpy.where(pending[:,1]< threshold)[0]

    #根据振幅分成两类
    '''
    features = pending[:,1]
    features = numpy.vstack(features)
    features = whiten(features)
    center, f = kmeans(features, 2)
    maxy = numpy.argmax(center)
    '''
    #逐个删除振幅小的点
    #todels = numpy.where(code != maxy)[0]

    id = 1
    loop = count-1
    while id<len(points)-1:
        dist = getDistance((points[id-1][0],points[id-1][1]),(points[id+1][0],points[id+1][1]),(points[id][0],points[id][1]))
        #type,_ = vq(numpy.array([[dist]]), center)
        #if type[0] != maxy:
        if dist<threshold:
            #遍历后续点，
            todel = id
            if id<len(points)-2:
                dist2 = getDistance((points[id][0],points[id][1]),(points[id+2][0],points[id+2][1]),(points[id+1][0],points[id+1][1]))
                #type,_ = vq(numpy.array([[dist2]]), center)
                #if type[0] != maxy:
                if dist2<dist:
                    todel = id+1
            #太小，删
            if todel == len(points)-1:
                break
            points = numpy.delete(points,todel,axis=0)
            continue
        id +=1

    #plot(points[:,0],points[:,1],'g+')
    #分隔
    i = 1
    count = points.shape[:2][0]
    left = 0
    res = []
    while i<count-1:
        if points[i][1] > points[i-1][1] and points[i][1] > points[i+1][1]:
            rightY = points[i+1][1]
            right = points[i+1][0]
            for x in range(i+1,count):
                if rightY>=points[x][1]:
                    rightY=points[x][1]
                    right=points[x][0]
                else:
                    break
            if(left>0):
                res += [[left-1,right-1]]
            else:
                res += [[left,right-1]]
            left = right
        i += 1
    if left<src.shape[:2][0]:
        res += [[left,src.shape[:2][0]]]
    return res

def findMinPos(arr,pos,reg):
    left = pos-reg;
    if left<0:
        left =0
    right = pos+reg;
    if right>len(arr):
        right = len(arr)

    values = arr[left:right].tolist()
    val = None;
    found=0;

    for i in range(left,right):
        if val is None:
            val = arr[i]
            found = i
            continue
        if arr[i]<val:
            val = arr[i]
            found = i;
        elif arr[i]==val:
            if i <= pos:
                val = arr[i]
                found = i;
            else:
                if pos-found>pos-i:
                    val = arr[i]
                    found = i;

    return found

def getSeparaters(src,orisum):
    src = signal.detrend(src)

    #找出拐点
    trend = False #False 下降，True上涨
    preval = 0 #上一个值
    points = [] #拐点
    pos = 0
    for i in src:
        if preval != i:
            if trend != (i>preval):
                trend = (i>preval)
                points += [[pos if pos == 0 else pos-1,preval,orisum[pos]]]
        pos = pos+1
        preval = i


    if len(points) <4:
        return [[0,len(src)]]
    points += [[pos,0]]

    #分隔
    i = 1
    points = numpy.array(points)
    count = points.shape[:2][0]
    left = 0
    res = []
    while i<count-1:
        if points[i][1] > points[i-1][1] and points[i][1] > points[i+1][1]:
            rightY = points[i+1][1]
            right = points[i+1][0]
            for x in range(i+1,count):
                if rightY>=points[x][1]:
                    rightY=points[x][1]
                    right=points[x][0]
                else:
                    break
            res += [[left,right]]
            left = right
        i += 1
    if left<src.shape[:2][0]:
        res += [[left,src.shape[:2][0]]]

    #搜索左右像素，调整分割点
    '''adjusted = []
    for i in res:
        l,r = i;
        adjusted.append([findMinPos(orisum,l,2),findMinPos(orisum,r,2)])'''
    return res

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
    r = min(x+w,x1+w1)

    t = max(y,y1)
    b = min(y+h,y1+h1)
    if l>=r or t>=b:
        return [0,0,0,0],False
    else:
        return [l,t,t-l,b-t],True


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

def uniteRectsOnce2(pos, contours,fuzzy,comp=None,leftOrRight=False):
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
        elif fuzzy:
            t = max(y,y1)
            b = min(y+h,y1+h1)
            if b>t and (b-t)>min(h,h1)/2.and (x1-x-w)<(float(max(h1,h))/5.):#左右相邻
                if comp and (w1<comp[2]) and leftOrRight:#右侧较大的,需要比较后确定和左合并还是和右侧合并
                    contours,did = uniteRectsOnce(i,contours,fuzzy)
                    if not did:
                        contours[pos] = joinRect((x,y,w,h),(x1,y1,w1,h1))
                        contours = numpy.delete(contours,i,axis=0)
                        return contours,True
                    else:
                        return contours,True

                elif float(min(w1,w))/float(max(w1,w))<0.5 and (min(w1,w)<max(h,h1)/2.):
                    contours,did = uniteRectsOnce(i,contours,fuzzy,[x,y,w,h],w1<w)
                    if not did:
                        contours[pos] = joinRect((x,y,w,h),(x1,y1,w1,h1))
                        contours = numpy.delete(contours,i,axis=0)
                        return contours,True
                    #else:
                    #    return contours,True
                #elif w1*2<w and w1<max(h1,h)/2:
                    #contours,did = uniteRectsOnce(i,contours)
                    #if not did:
                        #contours[pos] = joinRect((x,y,w,h),(x1,y1,w1,h1))
                        #contours = numpy.delete(contours,i,axis=0)
                        #return contours,True
                    #else:
                    #    return contours,True
            if x1>x+w:
                return contours,False
        i+=1
    return contours,False

class Row:

    def __init__(self):
        self.items = []
        self.top = None
        self.bottom = None
    def append(self, rc, adjust=True):

        ''' i = 0
        while i < len(self.items):
            #上下排序，合并
            p = self.items[i]
            _,intersert = intersectRect(rc,(p[0],rc[1],p[2],p[3]))
            if intersert:
                rc = joinRect(rc,p)
                self.items.pop(i)
                continue
            i+=1'''
        x,y,w,h = rc
        self.items += [rc]

        if adjust:
            if self.top is None:
                self.top = y
            else:
                self.top = (self.top+min(y,self.top))/2
            if self.bottom is None:
                self.bottom = y+h
            else:
                self.bottom = (self.bottom + max(y+h,self.bottom))/2

    def getDist(self,rc):
        x,y,w,h = rc
        if y+h<self.top:
            return y+h - self.top
        elif y>self.bottom:
            return y - self.bottom
        return 0

def findClosestRow(rows,rc):
    min = None
    found = None
    for i in range(0,len(rows)):
        if min is None:
            found = i
            min = rows[i].getDist(rc)
        else:
            dis = rows[i].getDist(rc)
            if(abs(dis) < abs(min)):
                min = dis
                found = i
    return found, min
#
def sortRects(contours,avgH):
    rows = []
    contours = contours[numpy.argsort(contours[:,0])]

    undt = [] #高度<avgH,待定
    for i in contours:
        if i[3]<avgH:
            undt += [i]
            continue
        if len(rows) == 0:
            r =  Row()
            r.append(i)
            rows += [r]
        else:
            ri,dist = findClosestRow(rows,i)
            if dist == 0:
                rows[ri].append(i)
            elif dist < 0:
                r = Row()
                r.append(i)
                rows.insert(ri,r)
            else:
                r = Row()
                r.append(i)
                rows.insert(ri+1,r)

    for i in undt:
        if len(rows) == 0:
            r = Row()
            r.append(i)
            rows += [r]
        else:
            ri,dist = findClosestRow(rows,i)
            if dist == 0:
                rows[ri].append(i,False)
            elif dist < 0:
                r = Row()
                r.append(i)
                rows.insert(ri,r)
            else:
                r = Row()
                r.append(i)
                rows.insert(ri+1,r)

    sorted = []
    for i in rows:
        items = numpy.array(i.items)
        items = items[numpy.argsort(items[:,0])]
        sorted += items.tolist()

    return numpy.array(sorted)

def separateWords2(features,count):
    centers, f = kmeans(features, count)
    centers = centers[:,1]
    centers = numpy.sort(centers)
    separates = []
    for pos in range(0,len(centers)-1):
        separates += [(centers[pos]+centers[pos+1])/2]
    return separates

def separateWords3(w,count):
    dis = float(w)/float(count)
    separates = []
    left = 0
    for pos in range(0,count-1):
        separates += [left +dis]
        left = dis+left

    return separates


def getWordsRect(src,recursive = True):

    mask = src.copy()
    #erode dilate

    mask = cv2.Canny(mask, 100, 200)
    #mask = mask + src
    _, contours, hierarchy  = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    rects=[]
    for i in contours:
        rc = cv2.boundingRect(i)
        #if rc[2]<5 and rc[3]<5:
        #    continue
        rects+=[rc]
    contours = numpy.array(rects)

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

    #最普遍的高度和宽度
    (n, bins) = numpy.histogram(candidates[:,1], bins=10, normed=True)
    mustH = bins[n.argmax(axis=0)]
    (n, bins) = numpy.histogram(candidates[:,0], bins=10, normed=True)

    pylab.hist(candidates, bins=10)
    mustW = bins[n.argmax(axis=0)]

    newcontours = []
    for i in contours:
        x,y,w,h = i
        if h > mustH*2: #高度过大的往往有问题
            continue
        else:
            newcontours += [i]

    contours = numpy.array(newcontours)


    #过宽的一般是分错了,再用kmeans分隔

    newcontours = []
    for i in contours:
        x,y,w,h = i
        if w<mustW*0.8:
            continue
        if w*mustH > mustW*h*1.5 and w> mustW:
            #continue

            clip = mask[y:y+h+1,x:x+w+1]
            #features = numpy.argwhere(clip>0).astype('float')

            selected = separateWords(clip)

            #selected = []
            #left = 0
            #while left<w:
            #    selected += [left]
            #    left += mustW
            left=0
            for pos in range(0,len(selected)-1):
                right = selected[pos]
                newcontours += [[x+left,y,right-left,h]]
                left = right+1
            newcontours += [[x+left,y,w-left,h]]
            continue
            #尝试分隔多次，取效果最好的一次
            clip = mask[y:y+h+1,x:x+w+1]
            #features = numpy.argwhere(clip>0).astype('float')

            trys = []
            trys  += [separateWords(clip)]
            '''if(max_label>1):
                trys  += [separateWords3(w,max_label)]
            #else:
            #    trys += [[w-1]]
            if(max_label>2):
                trys += [separateWords3(w,max_label-1)]
            #else:
            #    trys += [[w-1]]'''

            minsum = None; #取分隔线上像素最少的方案
            selected = None;
            for t in trys:
                line = clip[:,t]
                count = line.shape[:2][1]
                sum = numpy.sum(line)/count
                if minsum is None or sum< minsum:
                    minsum = sum
                    selected = t

            left = 0
            pos = 0
            for pos in range(0,len(selected)-1):
                right = (selected[pos]+selected[pos+1])/2
                newcontours += [[x+left,y,right-left,h]]
                left = right+1
            newcontours += [[x+left,y,w-left,h]]
        else:
            newcontours += [i]
    contours = numpy.array(newcontours)

    '''added = []
    #宽度过大
    pos = 0
    while pos < len(contours):
        x,y,w,h=contours[pos]
        if w>=2*avgW:
            words = separateWords(src[y:y+h+1,x:x+w+1])
            for i in words:
                i[0] += x
                i[1] += y

            if len(words):
                contours = numpy.delete(contours,pos,axis=0)
                added += words
                continue
        pos += 1

    if len(added):
        contours = numpy.append(contours,added,axis=0)'''
    contours = sortRects(contours,avgH)

    #合并相近的矩形
    '''pos = 1
    while pos < len(contours)-1:
        x0,y0,w0,h0 = contours[pos-1]
        x,y,w,h = contours[pos]

        x1,y1,w1,h1 = contours[pos+1]
        if w<avgW/2.0:
            if abs(x-x0-w0) <abs(x1-x-w) and abs(x-x0-w0)<avgW/5.0:
                #pass
                contours[pos-1] = joinRect((x0,y0,w0,h0),(x,y,w,h))
                contours = numpy.delete(contours,pos,axis=0)
                pos -= 1
                continue
            elif abs(x1-x-w)<avgW/5.0:
                #pass
                contours[pos] = joinRect((x1,y1,w1,h1),(x,y,w,h))
                contours = numpy.delete(contours,pos+1,axis=0)
                continue
        pos += 1'''

    #删除过小的矩形
    pos = 0
    while pos < len(contours):
        x,y,w,h=contours[pos]
        if h<mustH*0.6:
            contours = numpy.delete(contours,pos,axis=0)
            continue
        pos += 1

    return contours



def separateWords(src):
    #tmp = cv2.dilate(src,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3)) )
    #tmp = closing(src, square(5))
    #selem = disk(10)
    #tmp = rank.mean(src, selem=selem)
    #tmp = rank.mean_percentile(src, selem=selem, p0=.1, p1=.9)
    #tmp = rank.mean_bilateral(src, selem=selem, s0=500, s1=500)


    '''figure()
    subplot(211)
    imshow(src)
    subplot(212)
    imshow(tmp)
    show()'''
    orisum = src.sum(axis=0)/255.0
    sum = filters.gaussian_filter(orisum,8)
    sep = getSeparaters(sum,filters.gaussian_filter(orisum,2))

    words = []
    pos = 0
    h = len(src)
    for i in sep:
        #words += [[i[0],0,i[1]-i[0],h]]
        word = src[:,i[0]:i[1]]
        x,y,w,h = bounding_box(word)
        words.append([i[0]+x,y,w,h])
        #if word is not None:
       #     words += [word]
    return words


