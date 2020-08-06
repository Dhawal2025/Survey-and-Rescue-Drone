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
import time

class sr_scheduler():

	def __init__(self):
		rospy.Subscriber('/detection_info',SRInfo,self.detection_callback)	
		rospy.Subscriber('/serviced_info',SRInfo,self.serviced_callback)
		self.decision_pub = rospy.Publisher('/decision_info',SRInfo,queue_size=4)
		self.supply={"MEDICINE":3,"FOOD":3,"RESCUE":1}
		#self.state={"RESCUE":[[]],"MEDICINE":[[]],"FOOD":[[]]}
		self.state={"RESCUE":[],"MEDICINE":[],"FOOD":[]}
		self.time_av={"RESCUE":5,"MEDICINE":27,"FOOD":27}
		self.current_tgt=""
		self.base=False
		self.start=False
		self.decided_msg = SRInfo()
		self.loc_base="C4"
		self.decided_msg.location = self.loc_base
		self.decided_msg.info = "BASE"
		self.current_state=""
		
	def detection_callback(self, msg):
		self.state[msg.info].append([msg.location,time.time()])
		print(self.state[msg.info])
		if(msg.info == "RESCUE" and self.current_state!= "RESCUE"):
			self.decided_msg.location = msg.location
			self.decided_msg.info= msg.info
			self.base = True
			self.current_state = "RESCUE"
			self.decision_pub.publish(self.decided_msg)
			self.state["RESCUE"].pop(0)
		elif self.current_tgt =="":
			self.pub()
		return
		

	def serviced_callback(self,msg):
		# Take appropriate action when either service SUCCESS or FAILIURE is recieved from monitor.pyc
		# if self.start==True and msg.info=="SUCCESS":
		# 	print("start successful")
		# 	self.supply["MEDICINE"]=3
		# 	self.supply["FOOD"]=3
		# 	self.current_tgt=""
		# 	self.start=False
		# 	self.pub()
		if msg.info == "FAILURE":
			print(msg.location+ "Faliure")
			return
		if self.base == True and msg.info == "SUCCESS":
			print("base successful")
			self.current_tgt=self.loc_base
			self.decided_msg.location=self.loc_base
			self.decided_msg.info="BASE"
			self.base = False
			self.supply["MEDICINE"]=3
			self.supply["FOOD"]=3
			self.decision_pub.publish(self.decided_msg)
		# else if self.current_tgt==self.loc_base and msg.info=="SUCCESS":
		# 	self.current_tgt=""
		else :
			if self.current_state=="FOOD":
				self.supply["FOOD"]-=1
			elif self.current_state=="MEDICINE":
				self.supply["MEDICINE"]-=1
			self.current_tgt=""
			self.current_state =""
			self.pub()
		return 

	def pub(self):
		if ( len(self.state["RESCUE"])==0 and len(self.state["MEDICINE"])==0 and len(self.state["FOOD"])==0 ):
			self.decided_msg.location=self.loc_base
			self.decided_msg.info = "BASE"
			self.current_tgt=""
			self.current_state=""
			print("at "+self.decided_msg.location)
			self.decision_pub.publish(self.decided_msg)
			self.supply["MEDICINE"]=3
			self.supply["FOOD"]=3
			return 

		for x in self.state:
			if len(self.state[x])>0 and self.current_tgt=="":
				self.state[x]=sorted(self.state[x],key=lambda x: x[1])
				if (x != "RESCUE") and (self.supply["MEDICINE"]==0 or self.supply["FOOD"]==0):
					self.supply["MEDICINE"]=3
					self.supply["FOOD"]=3
					print("refill")
					break
					# if (self.supply["MEDICINE"] == 0 and len(self.state["MEDICINE"])>0) and (self.supply["FOOD"] == 0 and len(self.state["FOOD"])>0): 
				for i in self.state[x]:
					if (self.time_av[x]>(time.time()-i[1]) and self.supply[x]>0):
						# if(i[0]=="A2"): #or i[0]=="D4"):
						# 	self.state[x].pop(0)
						# 	continue
						self.decided_msg.location = i[0]
						self.current_tgt=i[0]
						self.current_state = x
						self.decided_msg.info = x
						if x=="RESCUE":
							self.base=True
						# print(self.decided_msg.location)
						self.decision_pub.publish(self.decided_msg)
						self.state[x].pop(0)
						return 
		return


	def shutdown_hook(self):
		# This function will run when the ros shutdown request is recieved.
		# For instance, when you press Ctrl+C when this is running
		pass



def main(args):
	
	sched = sr_scheduler()
	rospy.init_node('sr_scheduler', anonymous=False)
	rospy.on_shutdown(sched.shutdown_hook)
	rate = rospy.Rate(30)
	# sched.decided_msg.location=sched.loc_base
	# sched.decided_msg.info="BASE"
	# sched.start=True
	# sched.current_tgt=sched.loc_base
	sched.decision_pub.publish(sched.decided_msg)
	while not rospy.is_shutdown():
		rate.sleep()

if __name__ == '__main__':
    main(sys.argv)