#!/usr/bin/env python

'''

This python file runs a ROS-node of name drone_control which holds the position of e-Drone on the given dummy.
This node publishes and subsribes the following topics:

		PUBLICATIONS			SUBSCRIPTIONS
		/drone_command			/whycon/poses
		/alt_error				/pid_tuning_altitude
		/pitch_error			/pid_tuning_pitch
		/roll_error				/pid_tuning_roll



Rather than using different variables, use list. eg : self.setpoint = [1,2,3], where index corresponds to x,y,z ...rather than defining self.x_setpoint = 1, self.y_setpoint = 2
CODE MODULARITY AND TECHNIQUES MENTIONED LIKE THIS WILL HELP YOU GAINING MORE MARKS WHILE CODE EVALUATION.
'''

# Importing the required libraries

from edrone_client.msg import *
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Int16
from std_msgs.msg import Int64
from std_msgs.msg import Float64
from pid_tune.msg import PidTune
import rospy
import time
import json
from survey_and_rescue.msg import *


class Edrone():
	"""docstring for Edrone"""

	def __init__(self):

		# initializing ros node with name drone_control
		rospy.init_node('drone_control')

		# This corresponds to your current position of drone. This value must be updated each time in your whycon callback
		# [x,y,z]
		self.drone_position = [0,0, 30]
		
		# [x_setpoint, y_setpoint, z_setpoint]
		# whycon marker at the position of the dummy given in the scene. Make the whycon marker associated with position_to_hold dummy renderable and make changes accordingly
		with open("/home/dhawal/catkin_ws/src/survey_and_rescue/scripts/cell_coords.json", 'rb') as input:
   				self.rect_list = json.load(input)
		self.setpoint=[]#self.rect_list["E4"]
		self.point =''

		# Declaring a cmd of message type PlutoMsg and initializing values
		self.cmd = edrone_msgs()
		self.cmd.rcRoll = 1500
		self.cmd.rcPitch = 1500
		self.cmd.rcYaw = 1500
		self.cmd.rcThrottle = 1500
		self.cmd.rcAUX1 = 1500
		self.cmd.rcAUX2 = 1500
		self.cmd.rcAUX3 = 1500
		self.cmd.rcAUX4 = 1500

		# initial setting of Kp, Kd and ki for [roll, pitch, throttle]. eg: self.Kp[2] corresponds to Kp value in throttle axis
		# after tuning and computing corresponding PID parameters, change the parameters
		self.Kp = [989*0.02, 1089*0.02,1900*0.06]
		self.Ki = [0.91,0.98, 0.5]
		self.Kd = [3100*0.3, 3800*0.3,3500*0.3]
		# self.Kp = [2751*0.02,2445*0.02,3100*0.06]
		# self.Ki = [0,               0, 13*0.008]
		# self.Kd = [4585*0.3, 4520*0.3, 4127*0.3]

		#-----------------------Add other required variables for pid here ----------------------------------------------#
		self.error = [0, 0, 0]
		self.prev_err = [0, 0, 0]
		self.iterm = [0, 0, 0]
		self.max_values = [1800,1800,1800]
		self.min_values = [1000, 1000, 1000]
		self.cumulative_time = 0  #variable to calculate cumulative time
		self.continuous_time=0 #variable to count continuous time
		self.comp=False
		# Hint : Add variables for storing previous errors in each axis, like self.prev_values = [0,0,0] where corresponds to [pitch, roll, throttle]		#		 Add variables for limiting the values like self.max_values = [2000,2000,2000] corresponding to [roll, pitch, throttle]
		#													self.min_values = [1000,1000,1000] corresponding to [pitch, roll, throttle]
		#																	You can change the upper limit and lower limit accordingly.
		# ----------------------------------------------------------------------------------------------------------

		# # This is the sample time in which you need to run pid. Choose any time which you seem fit. Remember the stimulation step time is 50 ms
		self.sample_time = 0.060  # in seconds

		# Publishing /drone_command, /alt_error, /pitch_error, /roll_error
		self.command_pub = rospy.Publisher('/drone_command', edrone_msgs, queue_size=1)
		self.k=0
		# ------------------------Add other ROS Publishers here-----------------------------------------------------
		self.alt_err = rospy.Publisher('/alt_error', Float64, queue_size=1)
		self.pitch_err = rospy.Publisher('/pitch_error', Float64, queue_size=1)
		self.roll_err = rospy.Publisher('/roll_error', Float64, queue_size=1)
		self.pos_err = rospy.Publisher('/pos_err', Float64, queue_size=1)
		self.half_pos_err = rospy.Publisher('/half_pos_err', Float64, queue_size=1)
		self.half_neg_err = rospy.Publisher('/half_neg_err', Float64, queue_size=1)
		self.neg_err = rospy.Publisher('/neg_error', Float64, queue_size=1)
		self.zero = rospy.Publisher('/zero', Float64, queue_size=1)
		# -----------------------------------------------------------------------------------------------------------

		# Subscribing to /whycon/poses, /pid_tuning_altitude, /pid_tuning_pitch, pid_tuning_roll
		rospy.Subscriber('whycon/poses', PoseArray, self.whycon_callback)
		rospy.Subscriber('/decision_info',SRInfo,self.setpoint_callback)
		rospy.Subscriber('/pid_tuning_altitude',PidTune, self.altitude_set_pid)
		# -------------------------Add other ROS Subscribers here----------------------------------------------------
		rospy.Subscriber('/pid_tuning_pitch', PidTune, self.pitch_set_pid)
		rospy.Subscriber('/pid_tuning_roll', PidTune, self.roll_set_pid)
        # ------------------------------------------------------------------------------------------------------------
		self.arm()  # ARMING THE DRONE# Disarming condition of the drone
	
	def disarm(self):
		self.cmd.rcAUX4 = 1100
		self.command_pub.publish(self.cmd)
		rospy.sleep(1)

    # Arming condition of the drone : Best practise is to disarm and then arm the drone.
	def arm(self):

		self.disarm()

		self.cmd.rcRoll = 1500
		self.cmd.rcYaw = 1500
		self.cmd.rcPitch = 1500
		self.cmd.rcThrottle = 1000
		self.cmd.rcAUX4 = 1500
		self.command_pub.publish(self.cmd)	# Publishing /drone_command
		rospy.sleep(1)


	# def time_count(self,setp):
	# 	self.interval=0.01
	# 	#print "time"
	# 	while self.cumulative_time<3.00:
	# 		if((self.drone_position[0]>=setp[0]-0.5 and self.drone_position[0]<=setp[0]+0.5) and (self.drone_position[1]>=setp[1]-0.5 and self.drone_position[1]<=setp[1]+0.5) and (self.drone_position[2]>=setp[2]-1.0 and self.drone_position[2]<=setp[2]+1.0)):
	# 			self.cumulative_time+=self.interval
	# 			self.continuous_time+=self.interval
	# 			time.sleep(self.interval)
	# 			print "continous_time=" + str(self.continuous_time)+ "      cumulative_time="+ str(self.cumulative_time) 
	# 		else:
	# 			self.continuous_time=0
	# 	if self.cumulative_time>=3.00:
	# 		self.comp =True
			

	# Whycon callback function
	# The function gets executed each time when /whycon node publishes /whycon/poses\
	def setpoint_callback(self,msg):
		print(msg.location)
		# if(self.k==4):
		# 	self.k=0
		# 	self.iterm[0] = self.iterm[0]/2
		# 	self.iterm[1] = self.iterm[1]/2
		# 	self.iterm[2] = self.iterm[2]/2
		# else:
		# 	self.k=self.k+1
		print(self.setpoint)
		self.setpoint=self.rect_list[msg.location]
	
	def whycon_callback(self,msg):
		self.drone_position[0] = msg.poses[0].position.x

		# --------------------Set the remaining co-ordinates of the drone from msg--------------------------------------------		   
		self.drone_position[1]=msg.poses[0].position.y
		self.drone_position[2]=msg.poses[0].position.z




		
		# ---------------------------------------------------------------------------------------------------------------



	# Callback function for /pid_tuning_altitude
	# This function gets executed each time when /tune_pid publishes /pid_tuning_altitude
	def altitude_set_pid(self,alt):
		self.Kp[2] = alt.Kp * 0.06 # This is just for an example. You can change the ratio/fraction value accordingly
		self.Ki[2] = alt.Ki * 0.008
		self.Kd[2] = alt.Kd * 0.3

	# ----------------------------Define callback function like altitide_set_pid to tune pitch, roll--------------
	def pitch_set_pid(self,pitch):
		self.Kp[1] = pitch.Kp * 0.02 # This is just for an example. You can change the ratio/fraction value accordingly
		self.Ki[1] = pitch.Ki * 0.008
		self.Kd[1] = pitch.Kd * 0.3

	def roll_set_pid(self,roll):
		self.Kp[0] = roll.Kp * 0.02 # This is just for an example. You can change the ratio/fraction value accordingly
		self.Ki[0] = roll.Ki * 0.008
		self.Kd[0] = roll.Kd * 0.3

















	# ----------------------------------------------------------------------------------------------------------------------


	def pid(self):
	# -----------------------------Write the PID algorithm here--------------------------------------------------------------

	# Steps:
	# 	1. Compute error in each axis. eg: error[0] = self.drone_position[0] - self.setpoint[0] ,where error[0] corresponds to error in x...
	#	2. Compute the error (for proportional), change in error (for derivative) and sum of errors (for integral) in each axis. Refer "Understanding PID.pdf" to understand PID equation.
	#	3. Calculate the pid output required for each axis. For eg: calcuate self.out_roll, self.out_pitch, etc.
	#	4. Reduce or add this computed output value on the avg value ie 1500. For eg: self.cmd.rcRoll = 1500 + self.out_roll. LOOK OUT FOR SIGN (+ or -). EXPERIMENT AND FIND THE CORRECT SIGN
	#	5. Don't run the pid continously. Run the pid only at the a sample time. self.sampletime defined above is for this purpose. THIS IS VERY IMPORTANT.
	#	6. Limit the output value and the final command value between the maximum(1800) and minimum(1200)range before publishing. For eg : if self.cmd.rcPitch > self.max_values[1]:
	#																														self.cmd.rcPitch = self.max_values[1]
	#	7. Update previous errors.eg: self.prev_error[1] = error[1] where index 1 corresponds to that of pitch (eg)
	#	8. Add error_sum


		# if((self.drone_position[0]>=self.setpoint[0]-0.5 and self.drone_position[0]<=self.setpoint[0]+0.5) and (self.drone_position[1]>=self.setpoint[1]-0.5 and self.drone_position[1]<=self.setpoint[1]+0.5) and (self.drone_position[2]>=self.setpoint[2]-1.0 and self.drone_position[2]<=self.setpoint[2]+1.0)):
		# 	self.cumulative_time+=0.033
		# 	print "At "+ str(self.point)+" for ="+ str(self.cumulative_time)
		# print(self.setpoint)
		if self.setpoint == []:
			return 
		self.error[0]=self.setpoint[0]-self.drone_position[0]
		self.error[1]=self.setpoint[1]-self.drone_position[1]
		self.error[2]=self.setpoint[2]-self.drone_position[2]
		if abs(self.error[0])>4 or abs(self.error[1])>4:
			#print "large error"
			self.error[0] = self.error[0]/5
			self.error[1] = self.error[1]/5
		self.iterm[0]=(self.iterm[0]+self.error[0])*self.Ki[1]
		self.iterm[1]=(self.iterm[1]+self.error[1])*self.Ki[0]
		self.iterm[2]=(self.iterm[2]+self.error[2])*self.Ki[2]
		self.out_roll = self.Kp[1]*self.error[0]+self.iterm[0]+self.Kd[1]*(self.error[0]-self.prev_err[0])
		self.out_pitch = self.Kp[0]*self.error[1]+self.iterm[1]+self.Kd[0]*(self.error[1]-self.prev_err[1])
		self.out_throttle = self.Kp[2]*self.error[2]+self.iterm[2]+self.Kd[2]*(self.error[2]-self.prev_err[2])
		self.cmd.rcRoll = 1500 - self.out_roll
		self.cmd.rcPitch = 1500 + self.out_pitch	
		self.cmd.rcThrottle = 1500 - self.out_throttle
		if self.cmd.rcRoll > self.max_values[0]:
			self.cmd.rcRoll = self.max_values[0]
		if self.cmd.rcRoll < self.min_values[0]:
			self.cmd.rcRoll = self.min_values[0]
		if self.cmd.rcPitch > self.max_values[1]:
			self.cmd.rcPitch = self.max_values[1]
		if self.cmd.rcPitch < self.min_values[1]:
			self.cmd.rcPitch = self.min_values[1]
		if self.cmd.rcThrottle > self.max_values[2]:
			self.cmd.rcThrottle = self.max_values[2]
		if self.cmd.rcThrottle < self.min_values[2]:
			self.cmd.rcThrottle = self.min_values[2]
		
		self.prev_err[0]=self.error[0]
		self.prev_err[1]=self.error[1]
		self.prev_err[2]=self.error[2]
		#print "r="+str(self.cmd.rcRoll)
		#print "p="+str(self.cmd.rcPitch)
		#print "t="+str(self.cmd.rcThrottle)






	# ------------------------------------------------------------------------------------------------------------------------


		
		self.command_pub.publish(self.cmd)
		self.alt_err.publish(self.error[2])
		self.pitch_err.publish(self.error[0])
		self.roll_err.publish(self.error[1])	
		self.neg_err.publish(-1.0)
		self.half_neg_err.publish(-0.5)
		self.half_pos_err.publish(0.5)
		self.pos_err.publish(1.0)
		self.zero.publish(0.0)		
	



if __name__ == '__main__':

	e_drone = Edrone()
	# wayp=[[-1.4,0.36,27.6],[-10.3,-5,27.8],[7.1,-5.2,29.2],[7.3,6.0,28.7],[-3.0,-1.4,28.2],[-1.4,-0.36,27.6]]
	start = time.time()
	r = rospy.Rate(30) #specify rate in Hz based upon your desired PID sampling time, i.e. if desired sample time is 33ms specify rate as 30Hz
	#d = json.load(open("/home/dhawal/catkin_ws/src/survey_and_rescue/scripts/cell_coords.json", 'r'))
	#print d
	# for f in wayp:#sorted(d.keys()):
		#f = #list([d[x][0],d[x][1],20.0])
		#f = d[f]
		# e_drone.point = f
		# e_drone.setpoint=f
		#print f
		#e_drone.comp = False
		# e_drone.cumulative_time=0.0
		#e_drone.continuous_time=0.0
		# while (not rospy.is_shutdown()) and (not e_drone.comp):
		# 	threading.Thread(target=e_drone.time_count,args=(f,)).start()
		#threading.Thread(target=e_drone.pid).start()
			#print e_drone.comp
		#while (not rospy.is_shutdown()) and (e_drone.cumulative_time<3.0):
	while not rospy.is_shutdown():
		e_drone.pid()
		r.sleep()
	#e_drone.disarm()
	#print time.time() - start
	print "Path Completed"

		
	
		