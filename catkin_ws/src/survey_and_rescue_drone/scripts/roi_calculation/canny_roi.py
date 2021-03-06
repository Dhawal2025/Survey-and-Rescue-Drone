#!/usr/bin/env python
from __future__ import print_function
import roslib
import sys
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import pickle
import imutils
import copy
import numpy as np
import itertools

class sr_determine_rois():

	def __init__(self):

		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber("/usb_cam/image_rect",Image,self.image_callback)
		self.img = None
		self.flag=0
	
	# CV_Bridge acts as the middle layer to convert images streamed on rostopics to a format that is compatible with OpenCV
	def image_callback(self, data):
		try:
			self.img = self.bridge.imgmsg_to_cv2(data, "bgr8")
			# cv2.imshow("image",self.img)
			# cv2.waitKey(1)
		except CvBridgeError as e:
			print(e)


	'''You will need to implement an image processing algorithm to detect the Regions of Interest (RoIs)
	The standard process flow is:
	i)		Standard pre-processing on the input image (de-noising, smoothing etc.)
	ii)		Contour Detection
	iii)	Finding contours that are square, polygons with 4 vertices
	iv)		Implementing a threshold on their area so that only contours that are the size of the cells remain'''
	def detect_rois(self):
		# Add your Code here
		# You may add additional function parameters
		# cv2.imshow("Detected ROIs", img_copy) #A copy is chosen because self.img will be continuously changing due to the callback function
		# cv2.waitKey(100)
		# self.gray= cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
		#print(self.img.shape)
		self.crop_img = self.img[6:950,140:1140]
		# self.resized = imutils.resize(self.crop_img, width=300)
		# self.ratio = self.crop_img.shape[0] / float(self.resized.shape[0])
		self.gray = cv2.cvtColor(self.crop_img,cv2.COLOR_BGR2GRAY)
		self.edged = cv2.Canny(self.gray, 30, 250)

		#print(self.gray.shape)
		#self.gray = self.gray[10:,20:280]
		# self.blurred = cv2.GaussianBlur(self.gray, (3, 5), 0)
		# self.mask = cv2.threshold(self.blurred, 50, 255, cv2.THRESH_BINARY_INV)[1]
		# #self.mask = cv2.inRange(self.blurred,0,50)
		# cnts = cv2.findContours(self.mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		# cnts = imutils.grab_contours(cnts)
		# print(len(cnts))
		# # self.mask = cv2.cvtColor(self.mask,cv2.COLOR_GRAY2BGR)
		#self.ch=0
		# self.matrix=[[]]
		# for c in cnts:
		# 	if len(c)>=4:
		# 		# #print(c)
		# 		(x, y, w, h) = cv2.boundingRect(c)
		# 		t= np.array((x, y, w, h))
		# 		if len(self.matrix)<=36:
		# 			self.matrix.append(t)
		# 		# ar = w / float(h)
		# 		# if ar>=0.95 and ar<=1.05:
		# 		# 	print(self.ch)
		# 		# 	self.ch+=1
		# 		# 	cv2.drawContours(self.mask, [c], -1, (0, 255, 0), 10)
		#  		t = t.astype("float")
		#  		t = t*self.ratio
		#  		t = t.astype("int")
		# 		cv2.rectangle(self.crop_img,(t[0],t[1]),(t[0]+t[2],t[1]+t[3]),(0,255,0),5)
		#  		#cv2.drawContours(self.crop_img, [c], -1, (0, 255, 0), 10)
		# filehandler = open("./roi.obj", 'w') 
		# pickle.dump(self.matrix, filehandler)
		cv2.namedWindow('img',cv2.WINDOW_NORMAL)
		cv2.resizeWindow('img', 600,600)
		# cv2.imshow("gray",self.gray)
		# cv2.imshow("mask",self.mask)
		cv2.imshow("img",self.edged)
		# self.flag=1
		#cv2.imwrite("result.jpg",self.gray)
		cv2.waitKey(1)
		#print("detect_rois")


	'''	Please understand the order in which OpenCV detects the contours.
		Before saving the files you may sort them, so that their order corresponds will the cell order
		This will help you greatly in the next part. '''
	def sort_rois(self):
		# Add your Code here
		print(not available)

		

	def query_yes_no(self, question, default=None):
		"""Ask a yes/no question via raw_input() and return their answer.

		"question" is a string that is presented to the user.
		"default" is the presumed answer if the user just hits <Enter>.
		It must be "yes" (the default), "no" or None (meaning
		an answer is required of the user).

		The "answer" return value is True for "yes" or False for "no".
		"""
		valid = {"yes": True, "y": True, "ye": True,"no": False, "n": False}
		if default is None:
			prompt = " [Y/N]:\t"
		elif default == "yes":
			prompt = " [Y/N]:\t"
		elif default == "no":
			prompt = " [Y/N]:\t"
		else:
			raise ValueError("Invalid default answer: '%s'" % default)

		while True:
			sys.stdout.write(question + prompt)
			choice = raw_input().lower()
			if default is not None and choice == '':
				return valid[default]
			elif choice in valid:
				return valid[choice]
			else:
				sys.stdout.write("\nPlease respond with 'yes' or 'no' ""(or 'y' or 'n').\n")

	'''	You may save the list using anymethod you desire
	 	The most starightforward way of doing so is staright away pickling the objects
		You could also save a dictionary as a json file, or, if you are using numpy you could use the np.save functionality
		Refer to the internet to find out more '''
	def save_rois(self):
		#Add your code here
		print("save_rois")

	#You may optionally implement this to display the image as it is displayed in the Figure given in the Problem Statement
	def draw_cell_names(self, img):
		#Add your code here
		print("hellocellnames")

def main(args):
	# try:
	rospy.init_node('sr_roi_detector', anonymous=False)
	r =	sr_determine_rois()
	while (not rospy.is_shutdown()):
		if r.img is not None:
			r.detect_rois()
			
	# cv2.waitKey(1)
	cv2.destroyAllWindows()
			# if('''No of cells detected is not 36'''):
			# 	new_thresh_flag = r.query_yes_no("36 cells were not detected, do you want to change ##Enter tweaks, this is not necessary##?")
			# 	if(new_thresh_flag):
			# 		#Change settings as per your desire
			# 		cv2.destroyAllWindows()
			# 	else:
			# 		continue
			# else:
			# 	satis_flag = r.query_yes_no("Are you satisfied with the currently detected ROIs?")
			# 	if(satis_flag):
			# 		#r.sort_rois()
			# 		#r.save_rois()
			# 		cv2.destroyAllWindows()
			# 		break
			# 	else:
			# 		print("hlkj")
					#Change more settings
					# r.draw_cell_names(r.img) # Again, this is optional
    # except KeyboardInterrupt:
	#     cv2.destroyAllWindows()
	#     print("end")


if __name__ == '__main__' :
	main(sys.argv)
	print("hello")