#!/usr/bin/env python
from __future__ import print_function
import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from survey_and_rescue.msg import *
from cv_bridge import CvBridge, CvBridgeError
import random
import json
import csv
import imutils
import copy

class sr_determine_colors():

	def __init__(self):
		self.detect_info_msg = SRInfo()
		self.bridge = CvBridge()
		self.img = None
		self.detect_pub = rospy.Publisher("/detection_info",SRInfo,queue_size=10) 
 		self.image_sub = rospy.Subscriber("/usb_cam/image_rect_color",Image,self.image_callback)
 		self.serviced_sub = rospy.Subscriber('/serviced_info',SRInfo,self.serviced_callback)
		#self.cells = ["A2","B3","C1","F2","D4","F6","C5","B6"]
		#self.state={"A2":"","B3":"","C1":"","F2":"","D4":"","F6":"","C5":"","B6":""}
		self.cells=[]
		self.state={}
 		with open("/home/dhawal/catkin_ws/src/survey_and_rescue/scripts/LED_Config.tsv") as tsvf:
			tsv = csv.reader(tsvf,delimiter="\t")
			for line in tsv:
				self.cells.append(line[0])
				self.state[line[0]]=""
		self.boundaries = [
		("RESCUE",[10, 10, 140], [105, 80, 255]),
		("MEDICINE",[86, 5, 4], [230, 88, 60]),
		("FOOD",[10, 105, 0], [90,255 , 80])]
		# self.boundaries = [
		# ("RESCUE",[17, 15, 100], [50, 56, 250]),
		# ("MEDICINE",[86, 31, 4], [220, 88, 50]),
		# ("FOOD",[30, 120, 30], [100,220 , 100])]

	def load_rois(self, file_path = "/home/dhawal/catkin_ws/src/survey_and_rescue/scripts/roi.json"):
		try:
			# s.rois = np.load("rois.npy")
			with open(file_path, 'rb') as input:
   				self.rect_list = json.load(input)
		except IOError, ValueError:
			print("File doesn't exist or is corrupted")


 	def image_callback(self, data):
 		try:
 			self.img = self.bridge.imgmsg_to_cv2(data, "bgr8")
			self.crop_img = self.img[128:980,275:1090]
 		except CvBridgeError as e:
 			print(e)


 	def serviced_callback(self, msg):
 		pass
 	
	def detect_color_contour_centers(self):
		l_r = np.array([10, 10, 140])
		h_r = np.array([90, 80, 255])
		l_b = np.array([86, 5, 4])
		h_b = np.array([230, 88, 60])
		l_g = np.array([10, 105, 10])
		h_g = np.array([90,250 , 70])
		mask_r = cv2.inRange(self.crop_img, l_r, h_r)
		output_r = cv2.bitwise_and(self.crop_img, self.crop_img, mask = mask_r)
		mask_g = cv2.inRange(self.crop_img, l_g, h_g)
		output_g = cv2.bitwise_and(self.crop_img, self.crop_img, mask = mask_g)
		mask_b = cv2.inRange(self.crop_img, l_b, h_b)
		output_b = cv2.bitwise_and(self.crop_img, self.crop_img, mask = mask_b)
		cv2.namedWindow('red',cv2.WINDOW_NORMAL)
		cv2.resizeWindow('red', 400,400)
		cv2.namedWindow('green',cv2.WINDOW_NORMAL)
		cv2.resizeWindow('green', 400,400)
		cv2.namedWindow('blue',cv2.WINDOW_NORMAL)
		cv2.resizeWindow('blue',400,400)
		cv2.imshow("red",mask_r)
		cv2.imshow("green",mask_g)
		cv2.imshow("blue",mask_b)
		cnts_r = cv2.findContours(mask_b.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		cnts_r = imutils.grab_contours(cnts_r)
		#c_r = max(cnts_r, key=cv2.contourArea)
		for i,c in enumerate(cnts_r):
			((x, y), radius) = cv2.minEnclosingCircle(c)
			print(str(i)+"-->"+str(radius))
		cv2.waitKey(0)

	def check_whether_lit(self):
		for cell in self.cells:
			for (i,lower,upper) in self.boundaries:
				lower = np.array(lower, dtype = "uint8")
				upper = np.array(upper, dtype = "uint8")
				self.roi = self.crop_img[self.rect_list[cell][1]:self.rect_list[cell][1]+self.rect_list[cell][3],self.rect_list[cell][0]:self.rect_list[cell][0]+self.rect_list[cell][2]]
				# cv2.rectangle(self.crop_img,(self.rect_list[cell][0],self.rect_list[cell][1]),(self.rect_list[cell][0]+self.rect_list[cell][2],self.rect_list[cell][0]+self.rect_list[cell][2]),(0,255,0),2)
				# cv2.imshow("imageh",self.roi)
				# cv2.waitKey(0)
				mask = cv2.inRange(self.roi, lower, upper)
				self.cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
				self.cnts = imutils.grab_contours(self.cnts)
				#print(len(cnts))
				#c = max(self.cnts, key=cv2.contourArea)
				#((x, y), radius) = cv2.minEnclosingCircle(c)
				if(len(self.cnts)>0):
					count =10
					while(count>0 and len(self.cnts)>0):
						mask = cv2.inRange(self.roi, lower, upper)
						self.cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
						self.cnts = imutils.grab_contours(self.cnts)
						count-=1
					cf = max(self.cnts,key=cv2.contourArea)
					((x,y),radius) = cv2.minEnclosingCircle(cf)
					if count==0 and self.state[cell]!=i and radius>14.0:	
						#print(cell+"--->"+i+"--->"+str(len(self.cnts)))
						#print(self.cnts)
						self.state[cell]=i
						self.detect_info_msg.location=cell
						self.detect_info_msg.info = i
						self.detect_pub.publish(self.detect_info_msg)
						break



def main(args):
	
	try:
		rospy.init_node('sr_beacon_detector', anonymous=False)
		s = sr_determine_colors()
		'''You may choose a suitable rate to run the node at.
		Essentially, you will be proceesing that many number of frames per second.
		Since in our case, the max fps is 30, increasing the Rate beyond that
		will just lead to instances where the same frame is processed multiple times.'''
		rate = rospy.Rate(5)
		# rate = rospy.Rate(5)
		s.load_rois()
		while s.img is None:
			pass
	except KeyboardInterrupt:
		cv2.destroyAllWindows()
	while not rospy.is_shutdown():
		try:
			#s.detect_color_contour_centers()
			s.check_whether_lit()
			rate.sleep()
		except KeyboardInterrupt:
			cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)