#!/usr/bin/env python
import roslib
import sys
import rospy
import cv2
import numpy as np
from matplotlib import pyplot as plt
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
class postion(object):
    def __init__(self):
        self.bridge_object = CvBridge()
        self.image_sub = rospy.Subscriber("/whycon/image_out",Image,self.camera_callback)
    
    def camera_callback(self,data):
        print("hello")
        try:
            print("tru")
            cv_image =self.bridge_object.imgmsg_to_cv2(data,desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)
        
        height, width, channels = cv_image.shape
        # descentre = 160
        # rows_to_watch = 20
        # crop_img = cv_image[(height)/2+descentre:(height)/2+(descentre+rows_to_watch)][1:width]
        gray= cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        # hist,bins = np.histogram(gray.flatten(),256,[0,256])
        # cdf = hist.cumsum()
        # cdf_normalized = cdf * float(hist.max()) / cdf.max()
        # plt.plot(cdf_normalized, color = 'b')
        # plt.hist(gray.flatten(),256,[0,256], color = 'r')
        # plt.xlim([0,256])
        # plt.legend(('cdf','histogram'), loc = 'upper left')
        # plt.show()
        # lower_yellow = np.array([20,100,100])
        # upper_yellow = np.array([50,255,255])
        # mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        # res = cv2.bitwise_and(crop_img,crop_img, mask= mask)
        # m = cv2.moments(mask, False)
        # try:
        #     cx, cy = m['m10']/m['m00'], m['m01']/m['m00']
        # except ZeroDivisionError:
        #     cy, cx = height/2, width/2
        # cv2.circle(res,(int(cx), int(cy)), 10,(0,0,255),-1)
        # cv2.imshow("Original", cv_image)
        # cv2.imshow("HSV", hsv)
        # cv2.imshow("MASK", mask)
        # cv2.imshow("RES", res)
        # mask1 = cv2.inRange(gray,0,70)
        # mask2 = cv2.inRange(gray,140,165)
        # cv2.namedWindow('gray',cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('gray', 600,600)
        # alpha=1.00
        # beta=1-alpha
        # dst = cv2.addWeighted(mask1, alpha, mask2, beta,0.0)
        # cv2.namedWindow('mask',cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('mask', 600,600)
        # clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(20,20))
        # dst = clahe.apply(gray)
        cv2.imshow("gray",dst)
        mask = cv2.inRange(dst,0,120)
        cv2.imshow("mask",mask)
        #cv2.imshow("mask",mask)
        cv2.waitKey(1)
        # error_x = cx-width /2
        # twist_object = Twist()
        # twist_object.linear.x=0.2
        # twist_object.angular.z = -error_x/100
        # rospy.loginfo("Angular value sent===>"+str(twist_object.angular.z))
        # self.movekobuki_object.move_robot(twist_object)
        
    
    def clean_up(self):
        cv2.destroyAllWindows()    

def main():

    rospy.init_node('line_following_node',anonymous=True)
    pos = postion()   
    rate = rospy.Rate(5)
    ctrl_c = False
    def shutdownhook():
    # works better than the rospy.is_shut_down()
        #pos.clean_up()
        cv2.destroyAllWindows()
        rospy.loginfo("shutdown time!")
        ctrl_c = True
    rospy.on_shutdown(shutdownhook)
    while not ctrl_c:
        rate.sleep()
    

if __name__ == '__main__':
    main()