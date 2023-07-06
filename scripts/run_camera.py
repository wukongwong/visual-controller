#!/usr/bin/env python
# license removed for brevity
import pyrealsense2 as rs
import numpy as np
import cv2
from cv_bridge import CvBridge
import time
import math
import rospy
from sensor_msgs.msg import Image

def talker():
    pub = rospy.Publisher('image', Image,queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(30) # 30hz

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # The code to set exposure time manually#
    profile = pipeline.start(config)
    # Get the sensor once at the beginning. (Sensor index: 1)
    sensor = pipeline.get_active_profile().get_device().query_sensors()[1]
    # Set the exposure anytime during the operation
    sensor.set_option(rs.option.exposure, 156.000)
    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Filename
    imageName1 = str(time.strftime("%Y_%m_%d_%H_%M_%S")) + '_Color.jpg'
    br = CvBridge()
    while not rospy.is_shutdown():
        print("Publishing image")
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        pub.publish(br.cv2_to_imgmsg(color_image))
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass



