from numpy import histogram

__author__ = 'caoym'


def removeBlack(src):
    #去除按色部分
    imhist, bins = histogram(src.flatten(),256,True)