import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
import pickle 

img = cv2.imread("/home/dhawal/catkin_ws/src/survey_and_rescue/scripts/result.jpg")
#img = img[0:650,20:850]
#blurred = cv2.GaussianBlur(img, (5, 5), 0)
#mask = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)[1]
resized = imutils.resize(img, width=300)
ratio = img.shape[0] / float(resized.shape[0])
# gray = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
# cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
# print(len(cnts))
# filehandler = open("./roi.obj", 'w') 
# pickle.dump(cnts, filehandler)
file = open("./roi.obj",'r')
cnts = pickle.load(file)
#cv2.imshow('image',img)
#print(img.shape)
# for c in cnts:
c = cnts[11].astype("float")
c *= ratio
c = c.astype("int")
cv2.namedWindow('mask',cv2.WINDOW_NORMAL)
cv2.resizeWindow('mask', 600,600)
cv2.drawContours(img, [c], -1, (0, 255, 0), 10)
		# ar = w / float(h)
		# if ar>=0.95 and ar<=1.05:
		# 	cv2.drawContours(img, [c], -1, (0, 255, 0), 10)
		# 	print(c)
cv2.imshow("mask",img)
# clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(20,20))
# dst = clahe.apply(img)
# mask = cv2.inRange(dst,0,35)
# cv2.imshow("mask",mask)
cv2.waitKey(0)
cv2.destroyAllWindows()