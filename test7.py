__author__ = 'caoym'
import cv2
import numpy as np
#import pdb
#pdb.set_trace()#turn on the pdb prompt

#read image
img = cv2.imread('D:\\data\\11111.jpg',cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('origin',img);

#SIFT
detector = cv2.SURF()
keypoints = detector.detect(gray,None)
img = cv2.drawKeypoints(gray,keypoints)
#img = cv2.drawKeypoints(gray,keypoints,flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('test',img);
cv2.waitKey(0)
cv2.destroyAllWindows()