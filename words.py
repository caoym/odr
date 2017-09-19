# coding=utf-8
from scipy import signal, ndimage
import cv2
import math
import numpy
from scipy.ndimage import measurements, filters
from skimage.feature import hog
from skimage.filters import threshold_adaptive
from PIL import Image
from matplotlib.pyplot import *

def get_features_from_image(src, maxW=4.):
    '''
    :param src:
    :param maxW: 截取特征的最大宽度（以高度的倍数为单位，3倍差不多是3个字的宽度）
    :return:
    '''
    res = []

    lines = get_text_lines_from_image(src)

    # figure()
    # gray()
    # i = 1
    # for lineData in lines:
    #     subplot(20, 1, i)
    #     subplots_adjust(left=0, right=1)
    #     axis('off')
    #     i += 1
    #     imshow(lineData)
    # show()


    for lineData in lines:
        words = numpy.array(separate_words_from_image(lineData))
        averageH = numpy.average(words[:, 3]);

        for i in range(0, len(words)):
            startX, startY, startW, startH = words[i]
            passed = 0
            pos = i
            clipTp = startY
            clipBt = startY + startH
            while (len == 0 or passed + startW < averageH * maxW) and pos < len(words):
                x, y, w, h = words[pos]
                clipLt = startX
                clipRt = x + w
                clipBt = max(y + h, clipBt)
                clipTp = min(y, clipTp)

                data = lineData[clipTp:clipBt, clipLt:clipRt]
                #
                # data2 = Image.fromarray(data).resize((64,32),Image.BILINEAR)

                # data = skeletonize(data/255)*255
                # data = zhangSuen(data/255)*255
                # 删除连续的空白列
                data = erase_black(data)
                data = Image.fromarray(data).resize((96, 32), Image.BILINEAR)
                # data = numpy.array(data)
                # data = skeletonize(data/255)*255

                '''data = cv2.bitwise_not(numpy.array(data))
                data = threshold_adaptive(data, 40, offset=10)
                data = numpy.array(data,'uint8')*255
                data = cv2.bitwise_not(data)
                #data, distance = medial_axis(data, return_distance=True)
                data = skeletonize(data/255)*255
                data = Image.fromarray(data)'''
                # figure()
                # gray()
                # imshow(data2)
                # figure()
                # imshow(labels_open)
                # show()
                # objs = measurements.find_objects(labels_open,nbr_objects_open)

                '''figure()
                gray()
                imshow(data)
                show()'''

                # data = numpy.array(data,'uint8').tolist()
                # data = scaler.fit_transform(data)
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
    return res


def get_text_lines_from_image(src):
    '''
    将图片按文本单行切割
    :param src:
    :return:图片数组
    '''
    # 调整大小
    src = adjust_size(src)

    temp = src.copy()

    src = cv2.Canny(src, 100, 200)
    src, slope = adjust_slope(src)

    # src = cv2.erode(src,cv2.getStructuringElement(cv2.MORPH_CROSS,(1, 3)) )
    # src = cv2.dilate(src,cv2.getStructuringElement(cv2.MORPH_CROSS,(1, 3)) )

    # figure()
    # gray()
    # subplot(321)
    # title('Original')
    # imshow(temp)
    #
    # subplot(322)
    # title('Step 1 - Canny')
    # imshow(src)

    src = cv2.dilate(src, cv2.getStructuringElement(cv2.MORPH_RECT, (40, 3)))
    #subplot(323)
    #title('Step 2 - dilate(40, 3)')
    #imshow(src1)
    src = cv2.erode(src, cv2.getStructuringElement(cv2.MORPH_RECT, (40, 3)))
    #subplot(324)
    #title('Step 3 - erode(40, 3)')
    #imshow(src2)
    src = cv2.erode(src, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    #subplot(325)
    #title('Step 4 - dilate(5, 5)')
    #imshow(src3)
    src = cv2.dilate(src, cv2.getStructuringElement(cv2.MORPH_RECT, (6, 5)))
    #subplot(326)
    #title('Step 5 - dilate(6, 5)')
    #imshow(src4)

    src = 1 * (src > 64)
    labels_open, nbr_objects_open = measurements.label(src)

    # 调整水平
    block_size = 41
    temp = threshold_adaptive(temp, block_size, offset=10)
    temp = numpy.array(temp, 'uint8') * 255
    #反转颜色(黑白)
    temp = cv2.bitwise_not(temp)

    # figure()
    # gray()
    # subplot(221)
    # title("Original")
    # imshow(temp.copy())


    if slope != 0:
        temp = ndimage.rotate(temp, slope)
        # 旋转后像素会平滑，重新二值化
        temp = cv2.bitwise_not(temp)
        temp = threshold_adaptive(temp, block_size, offset=20)
        temp = numpy.array(temp, 'uint8') * 255
        temp = cv2.bitwise_not(temp)

        #subplot(222)
        #title("Rotated")
        #imshow(temp)
    # show()
    lines = [];



    # image = numpy.zeros(numpy.array(src).shape)
    count = 0
    for i in range(1, nbr_objects_open + 1):

        test = temp.copy()
        test[labels_open != i] = 0
        box = bounding_box(test)
        x, y, w, h = box
        if h < 10 or w < 3:
            continue;
        # 忽略靠近上下边的区域
        '''if y<2:
            continue
        if y+h> len(temp)-2:
            continue'''
        data = test[y:y + h, x:x + w]
        lines.append(data)

        copy = src.copy() * 255.
        copy[labels_open != i] = 0
        box = bounding_box(copy)
        x, y, w, h = box

        # 真对 E 形的图像（上下两行或者多行间，有连接），先做侵蚀，在膨胀，可以遍分割为多个区域
        toerode = w / 3
        if toerode <= 1:
            continue

        copy = cv2.erode(copy, cv2.getStructuringElement(cv2.MORPH_RECT, (toerode, 1)))
        copy = cv2.dilate(copy, cv2.getStructuringElement(cv2.MORPH_RECT, (toerode, 1)))
        copy = 1 * (copy > 128)

        sub_labels_open, sub_nbr_objects_open = measurements.label(copy)
        if (sub_nbr_objects_open > 1):
            for i in range(1, sub_nbr_objects_open + 1):
                test = temp.copy()
                test[sub_labels_open != i] = 0
                box = bounding_box(test)
                # count+=1
                # image[sub_labels_open == i] = count
                x, y, w, h = box
                if h < 10 or w < 3:
                    continue;
                # 忽略靠近上下边的区域
                if y < 2:
                    continue
                if y + h > len(temp) - 2:
                    continue

                data = test[y:y + h, x:x + w]
                lines.append(data)

    return lines


def get_separaters_from_image(src, orisum):
    '''
    从单行文本图片中获取每个字之间的切割位置
    :param src:
    :param orisum:
    :return:
    '''

    src = signal.detrend(src)

    # 找出拐点
    trend = False  # False 下降，True上涨
    preval = 0  # 上一个值
    points = []  # 拐点
    pos = 0
    for i in src:
        if preval != i:
            if trend != (i > preval):
                trend = (i > preval)
                points += [[pos if pos == 0 else pos - 1, preval, orisum[pos]]]
        pos = pos + 1
        preval = i

    if len(points) < 4:
        return [[0, len(src)]]
    points += [[pos, 0]]

    # 分隔
    i = 1
    points = numpy.array(points)
    count = points.shape[:2][0]
    left = 0
    res = []
    while i < count - 1:
        if points[i][1] > points[i - 1][1] and points[i][1] > points[i + 1][1]:
            rightY = points[i + 1][1]
            right = points[i + 1][0]
            for x in range(i + 1, count):
                if rightY >= points[x][1]:
                    rightY = points[x][1]
                    right = points[x][0]
                else:
                    break
            res += [[left, right]]
            left = right
        i += 1
    if left < src.shape[:2][0]:
        res += [[left, src.shape[:2][0]]]

    # 搜索左右像素，调整分割点
    '''adjusted = []
    for i in res:
        l,r = i;
        adjusted.append([findMinPos(orisum,l,2),findMinPos(orisum,r,2)])'''
    return res


def separate_words_from_image(src):
    '''
    从单行文本图片中切割出每一个字
    :param src:
    :return:
    '''
    '''figure()
    subplot(211)
    imshow(src)
    subplot(212)
    imshow(tmp)
    show()'''

    #subplot(311)
    #axis('off')
    #imshow(src)

    orisum = src.sum(axis=0) / 255.0

    #plot(orisum*2-80,'b-')

    sum = filters.gaussian_filter(orisum, 8)
    #plot(sum*2-40, 'g-')

    sep = get_separaters_from_image(sum, filters.gaussian_filter(orisum, 2))

    # separates = numpy.array(sep)[:,1]-1
    # graph = numpy.zeros( (src.shape[1], 1) )
    # graph[separates] = 32
    # plot(graph,  'r-')
    #
    # show()

    words = []
    pos = 0
    h = len(src)
    for i in sep:
        # words += [[i[0],0,i[1]-i[0],h]]
        word = src[:, i[0]:i[1]]
        x, y, w, h = bounding_box(word)
        words.append([i[0] + x, y, w, h])
        # if word is not None:
        #     words += [word]
    return words


def erase_black(src):
    '''
    剪切掉图片的纯黑色边框
    :param src:
    :return:
    '''
    i = 0
    while True:
        h, w = src.shape
        if i >= w:
            break;

        if (numpy.max(src[:, i:i + 1]) < 128):
            if i == w - 1 or numpy.max(src[:, i + 1:i + 2]) < 128:
                src = numpy.delete(src, i, axis=1)
                continue
        i += 1
    return src


def bounding_box(src):
    '''
    矩阵中非零元素的边框
    :param src:
    :return:
    '''
    B = numpy.argwhere(src)
    if B.size == 0:
        return [0, 0, 0, 0]
    (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1

    return [xstart, ystart, xstop - xstart, ystop - ystart]


def get_lines_from_image(src):
    '''
    从图片中拟合直线
    :param src:
    :return:
    '''
    # srcArr = numpy.array(src, 'uint8')
    # srcArr,T = rof.denoise(srcArr, srcArr)
    dst = cv2.Canny(src, 50, 200)
    lines = cv2.HoughLinesP(dst, 2, math.pi / 180.0, 40, numpy.array([]), 50, 10)[0]
    if lines is None:
        return None
    res = []
    for line in lines:
        x = (line[2] - line[0])
        y = (line[3] - line[1])
        xy = (x ** 2. + y ** 2.) ** 0.5
        if 0 == xy:
            continue
        sin = y / xy
        angle = numpy.arcsin(sin) * 360. / 2. / numpy.pi

        res += [[line[0], line[1], line[2], line[3], 1, sin, angle]]

    return numpy.array(res)


def show_lines_over_image(img, lines):
    """
    画出检查到的线
    :param img:
    :param lines:
    :return:
    """
    figure()
    gray()
    imshow(img)
    for line in lines:
        plot([line[0], line[2]], [line[1], line[3]], 'r-')

    show()


def adjust_slope(src):
    """
    矫正图片角度
    :type src: numpy.array
    """
    h, w = src.shape[:2]
    lines = get_lines_from_image(src)

    if lines is None:
        return src, 0
    #画出检查到的线
    # figure()
    # gray()
    # imshow(src)
    # for line in lines:
    #    plot([line[0], line[2]],[line[1], line[3]],'r-')
    # show()
    # bins = len(lines)/5
    # n, bins = numpy.histogram(lines[:,6], bins=180, normed=True)
    # plot(lines[:, 6])
    # hSlope = bins[n.argmax(axis=0)]
    angle = numpy.median(lines[:, 6])
    if abs(angle) < 0.1:
        angle = 0
        dest = src
    else:
        dest = ndimage.rotate(src, angle)

    return dest, angle


def adjust_size(src):
    """
    矫正图片大小
    通过一张图片中拟合到的直线长度与图片长宽的比例，确定图片的大小是否合适
    :type src: numpy.array
    """
    for loop in range(0, 8):
        h, w = src.shape[:2]
        lines = get_lines_from_image(src)

        #show_lines_over_image(src, lines)
        if lines is None:
            return src
        # 左右最大间距
        left = lines[:, 0]
        left = sorted(left, reverse=False)
        cut = len(left) / 3 + 1
        left = numpy.median(numpy.array(left)[:cut])
        right = lines[:, 2]
        right = sorted(right, reverse=True)
        cut = len(right) / 3 + 1
        right = numpy.median(numpy.array(right)[:cut])
        maxlen = right - left
        # 平均宽度
        arvlen = []
        for line in lines:
            arvlen += [line[2] - line[0]]

        arvlen = numpy.median(arvlen)
        if maxlen > arvlen * 8:
            pil_im = Image.fromarray(numpy.uint8(src))
            src = numpy.array(pil_im.resize((int(w * 0.8), int(h * 0.8)), Image.BILINEAR))
        else:
            break
    return src


def visualization_features(src, maxW=4.):
    figure()
    gray()
    features = get_features_from_image(src, maxW)

    i = 0
    for img in features:
        #canny = cv2.Canny(numpy.uint8(img), 50, 200)
        #desc, hog_image = hog(canny, orientations=6, pixels_per_cell=(4, 4), cells_per_block=(2, 2), visualise=True)
        # figure()
        # gray()
        # subplot(221)
        # imshow(img)
        # subplot(222)
        # imshow(hog_image)
        # show()
        # continue

        subplot(20, 20, i + 1)
        axis('off')
        imshow(img)
        i += 1
        if i == 400:
            figure()
            show()
            i = 0
    show()


def show_image(img):
    """
    显示图片
    :param img:
    :return:
    """
    figure()
    gray()
    imshow(img)
    show()
