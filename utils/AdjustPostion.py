# coding=utf-8

__author__ = 'caoym'

import math
from matplotlib.pyplot import plot, figure, show, imshow, gray
import numpy
from scipy import ndimage
from scipy.cluster.vq import vq
import homography
from PIL import Image
from scipy.cluster.vq import *
from scipy import ndimage
import cv2


def getLines(src):
    #srcArr = numpy.array(src, 'uint8')
    # srcArr,T = rof.denoise(srcArr, srcArr)
    dst = cv2.Canny(src, 50, 200)
    lines = cv2.HoughLinesP(dst, 2, math.pi/180.0, 40, numpy.array([]), 50, 10)
    if lines is None:
        return None
    res = []
    for line in lines:
        x = (line[0][2] - line[0][0])
        y = (line[0][3] - line[0][1])
        xy = (x ** 2 + y ** 2) ** 0.5
        if 0 == xy:
            continue
        sin = y / xy
        angle = numpy.arcsin(sin) * 360 / 2 / numpy.pi

        res += [[line[0][0], line[0][1], line[0][2], line[0][3], 1, sin, angle]]

    '''figure()
    gray()
    imshow(src)
    for line in res:
        plot([line[0], line[2]],[line[1], line[3]],'r-')
    show()'''
    return numpy.array(res)

# 分类
def classify(features):
    features = numpy.vstack(features)
    center, f = kmeans(features, 2)
    code, dis = vq(features, center)
    nd0 = numpy.where(code == 0)[0]
    nd1 = numpy.where(code == 1)[0]

    return nd0, nd1, center


# 回归
def polyfitFun(x, y, deg):
    z1 = numpy.polyfit(x, y, deg)
    return numpy.poly1d(z1)


# 奇异值分解降噪
def denoiseWithSVD(src):
    U, s, V = numpy.linalg.svd(src, full_matrices=False)
    s[1:] = 0
    S = numpy.diag(s)
    return numpy.dot(U, numpy.dot(S, V))


def adjustSize(src):
    """
    :type src: numpy.array
    """
    for loop in range(0,8):
        h, w = src.shape[:2]
        lines = getLines(src)
        if lines is None:
            return src
        #左右最大间距
        left = lines[:,0]
        left = sorted(left, reverse=False)
        cut = len(left)/3+1
        left = numpy.median(numpy.array(left)[:cut])
        right = lines[:,2]
        right = sorted(right, reverse=True)
        cut = len(right)/3+1
        right = numpy.median(numpy.array(right)[:cut])

        maxlen = right - left

        #平均宽度
        arvlen = []
        for line in lines:
            arvlen += [line[2] - line[0]]

        arvlen = numpy.median(arvlen)
        if maxlen>arvlen*8:
            pil_im = Image.fromarray(numpy.uint8(src))
            src = numpy.array(pil_im.resize((int(w*0.8),int(h*0.8)) ,Image.BILINEAR))
        else:
            break

    return src

def adjust(src):

    """
    :type src: numpy.array
    """
    for loop in range(0,8):
        h, w = src.shape[:2]
        lines = getLines(src)
        ws = lines[:,2] - lines[:,0]
        ws[(ws<0)] = -ws[(ws<0)]
        hs = lines[:,3] - lines[:,1]
        hs[(hs<0)] = -hs[(hs<0)]

        mustW = numpy.max(ws)
        mustH = numpy.max(hs)

        #(n, bins) = numpy.histogram(ws, bins=20, normed=True)
        #mustW = bins[n.argmax(axis=0)]
        #(n, bins) = numpy.histogram(hs, bins=20, normed=True)
        #mustH = bins[n.argmax(axis=0)]

        if mustW>w/4 or mustH > h/4 or w<300 or h<100:
            break
        else:
            pil_im = Image.fromarray(numpy.uint8(src))
            src = numpy.array(pil_im.resize((w/2,h/2)))


    if lines is None:
        return src
    # 画出检查到的线
    #figure()
    #gray()
    #imshow(src)
    #for line in lines:
    #    plot([line[0], line[2]],[line[1], line[3]],'r-')
    #bins = len(lines)/5
    n, bins = numpy.histogram(lines[:,6], bins=180, normed=True)
    hSlope = bins[n.argmax(axis=0)]

    dest = ndimage.rotate(src, hSlope)
    return dest

def adjustSlope(src):

    """
    :type src: numpy.array
    """
    h, w = src.shape[:2]
    lines = getLines(src)

    if lines is None:
        return src,0
    # 画出检查到的线
    #figure()
    #gray()
    #imshow(src)
    #for line in lines:
    #    plot([line[0], line[2]],[line[1], line[3]],'r-')
    #bins = len(lines)/5
    #n, bins = numpy.histogram(lines[:,6], bins=180, normed=True)
    #hSlope = bins[n.argmax(axis=0)]
    hSlope = numpy.median(lines[:,6])
    if abs(hSlope)<3:
        hSlope = 0
        dest = src
    else:
        dest = ndimage.rotate(src, hSlope)
    return dest,hSlope

def adjust2(src):
    """
    :type src: numpy.array
    """
    h, w = src.shape[:2]
    lines = getLines(src)
    if lines is None:
        return src
    # 画出检查到的线
    #for line in lines:
    #    plot([line[0], line[2]],[line[1], line[3]],'r-')

    # 分类 区分垂直和水平线
    ndx, ndy, kms = classify(lines[:, (4, 6)])

    if abs(kms[0][1]) > 45:
        ndx, ndy = ndy, ndx
    '''
    plot(lines[ndx,4],lines[ndx,6],'r.')
    plot(lines[ndy,4],lines[ndy,6],'b.')
    plot(kms[:,0],kms[:,1],'go')

    # 坐标和角度的关系
    figure()
    imshow(src)

    for i in lines[ndx,:]:
       plot([i[0], i[2]],[i[1], i[3]],'r')
    for i in lines[ndy,:]:
        plot([i[0], i[2]],[i[1], i[3]],'g')

    figure()
    plot(lines[ndx,1], lines[ndx,6],'ro')
    '''
    # 回归，计算水平旋转与Y轴的关系
    pfH = polyfitFun(lines[ndx, 1], lines[ndx, 6], 1)
    '''
    plot(lines[ndx,1], pfH(lines[ndx,1]),'g.')
    # 奇异值分解降噪
    # temp = denoiseWithSVD(datas[ndx,:][:,(4,6)])
    # plot(datas[ndx,1], temp[:,1],'b.')

    # figure()
    '''
    # 垂直旋转与X轴的关系
    #pfV = polyfitFun(lines[ndy, 0], lines[ndy, 6], 1)
    '''
    plot(lines[ndy,0], lines[ndy,6],'b.')
    plot(lines[ndy,0], pfV(lines[ndy,0]),'b*')
    '''

    if False:  # 通过计算四个顶点的偏移，得到偏移后的四边形，进行仿射变换，映射到原始矩形上
        pA = [0, 0]
        pB = [w, 0]
        pC = [w, h]
        pD = [0, h]
        # 旋转4条边相交获取新的四个顶点
        pASlope = numpy.tan(pfH(pA[1]) * numpy.pi / 180)
        pBSlope = numpy.tan(pfV(pB[0]) * numpy.pi / 180)
        pCSlope = numpy.tan(pfH(pC[1]) * numpy.pi / 180)
        pDSlope = numpy.tan(pfV(pD[0]) * numpy.pi / 180)

        pAB = lines.line_intersect(pA, pB, pASlope, pBSlope)
        pBC = lines.line_intersect(pB, pC, pBSlope, pCSlope)
        pCD = lines.line_intersect(pC, pD, pCSlope, pDSlope)
        pDA = lines.line_intersect(pD, pA, pDSlope, pASlope)
        # figure()
        # imshow(src)
        # plot([pAB[0],pBC[0]],[pAB[1],pBC[1]])
        # plot([pBC[0],pCD[0]],[pBC[1],pCD[1]])
        # plot([pCD[0],pDA[0]],[pCD[1],pDA[1]])
        # plot([pDA[0],pAB[0]],[pDA[1],pAB[1]])
        # figure()

        dest = affine(src, (pDA, pAB, pBC, pCD), ([0, 0], [w, 0], [w, h], [0, h]))

        # imshow(dest)
        # dest = Image.fromarray(dest)
        # dest.save('D:\\cloud\\IMG_3360_a.JPG')
        return dest
    else:  # 简单根据中心点上水平线的倾斜，旋转转图片
        # figure()
        # 估算中心点的倾斜角度
        hSlope = pfH(h / 2)
        #dest = Image.fromarray(src).rotate(hSlope)
        dest = ndimage.rotate(src, hSlope)
        # imshow(dest)
        # dest.save('D:\\cloud\\IMG_3360_b.JPG')
    return dest

def affine2(img, lt, rt, rd, ld, w, h):
    warp_dst = Image.new(img.mode, img.size)

    dest = numpy.float32([[lt[0], lt[1]], [ld[0], ld[1]], [rd[0], rd[1]]])
    src = numpy.float32([[0, 0], [0, h], [w, h]])
    warp_mat = cv2.getAffineTransform(src, dest)

    cv2.warpAffine(img, warp_dst, warp_mat, warp_dst.size());
    return warp_dst


# 仿射变换
def affine(img, fromArea, toArea):  # fromArea, toArea):

    # affine(img,(),)
    from_lt, from_rt, from_rd, from_ld = fromArea
    to_lt, to_rt, to_rd, to_ld = toArea
    # 左上 角、 右上 角、 右下 角、 左下 角
    src = numpy.array([[from_lt[1], from_lt[0], 1], [from_rt[1], from_rt[0], 1], [from_rd[1], from_rd[0], 1],
                       [from_ld[1], from_ld[0], 1]])
    dest = numpy.array(
        [[to_lt[1], to_lt[0], 1], [to_rt[1], to_rt[0], 1], [to_rd[1], to_rd[0], 1], [to_ld[1], to_ld[0], 1]])

    # maxY = ( lt[1],  rt[1])
    # 仿射变换
    # 估算 单 应 矩阵 
    H = homography.H_from_points(dest.T, src.T)
    # 辅助 函数， 用于 进行 几何 变换 
    def warpfcn(x):
        x = numpy.array([x[0], x[1], 1])
        xt = numpy.dot(H, x)
        xt = xt / xt[2]
        return xt[0], xt[1]

    # 用 全 透视 变换 对 图像 进行 变换
    return ndimage.geometric_transform(img, warpfcn)
