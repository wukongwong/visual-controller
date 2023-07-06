#!/usr/bin/env python
# license removed for brevity
import rospy
import sys
sys.path.append('../src')
import torch
import PIL.Image as Image_PIL
import time
import numpy as np
from inference import VSNet_realtime
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

detection_flag = True

# normalization parameters
label_0_max = 0.18997402794856910191
label_1_max = 0.19000000000000000222
label_2_max = 0.00000000000000000000
label_3_max = 1.00000000000000000000

label_0_min = -0.18997402794856910191
label_1_min = -0.19000000000000000222
label_2_min = -0.28000000000000002665
label_3_min = 0.00000000000000000000

# obtian object bounding box using YOLOv5
# input: img
#output: seg with circle mask
model_yolo = torch.hub.load(repo_or_dir='/home/zls/king/yolov5', model='yolov5s', source='local', pretrained=True)



def get_object_bounding_box_yolov5(img):
    start_time = time.time()
    mask = np.zeros((480, 480), dtype="uint8")
    end_time = time.time()
    # print("mask_generare",end_time-start_time)

    # resize image to fit model
    img = img.resize((480, 480))
    start_time = time.time()
    # 進行物件偵測
    results = model_yolo(img)
    end_time = time.time()
    # print("YOLO_infer_time",end_time-start_time)
    print(results.xyxy[0].cpu().numpy()[:, 5], "class")

    if len(results.xyxy[0].cpu().numpy())==0:
        detection_flag = False
        print("Detected noting")
        return img_goal, detection_flag
    #顯示結果摘要
    results.print()
    print(results.xyxy[0].cpu().numpy())

    # calculate the outer tangent circle from two diagonal points
    center_x = (results.xyxy[0].cpu().numpy()[0][0]+results.xyxy[0].cpu().numpy()[0][2])/2
    center_y = (results.xyxy[0].cpu().numpy()[0][1]+results.xyxy[0].cpu().numpy()[0][3])/2
    center_z = 1
    detection_flag = True
    # calculate the radius of the mask
    radius = ((results.xyxy[0].cpu().numpy()[0][0]-center_x)**2+(results.xyxy[0].cpu().numpy()[0][1]-center_y)**2)**0.5
    start_time = time.time()
    cv2.circle(mask, (int(center_x), int(center_y)), int(radius), 255, -1)
    end_time = time.time()
    # print("circle_time_*",end_time-start_time)
    # single channel to 3 channel
    # cv2.imshow("mask", mask)
    mask_ = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    # img = Image_PIL.fromarray(mask, 'RGB')
    # show the mask_
    # cv2.imshow("mask", mask_)

    return mask_, detection_flag


#caculate mask area
def caculate_mask_area(seg):
    area = 0
    [h, w] = np.shape(seg)
    # print(h,w)
    for i in range(h):
        for j in range(w):
            if seg.getpixel((i,j)) != 0:
                area += 1
    mask_ratio = area/(h*w)
    # print(mask_ratio,'mask_ratio')
    return area,mask_ratio

# img_goal = '/home/zls/Desktop/2000.png'
img_goal = '/home/zls/Desktop/goal_linear.png'
def img2control(img):
    start_time = time.time()
    img_current_mask,is_detected = get_object_bounding_box_yolov5(img)
    end_time = time.time()
    # print("YOLO_whole_time",end_time-start_time)

    # Waits for everything to finish running
    start_time = time.time()
    if is_detected:
        output = net.infer(img_current_mask, img_goal)
    else:
        output = [-label_0_min/(label_0_max-label_0_min),-label_1_min/(label_1_max-label_1_min),
                  -label_2_min/(label_2_max-label_2_min),1]
    end_time = time.time()
    return output


model_dir = '/media/zls/My Passport/save_linear_best/VSNet-train/'
model = model_dir + 'model.pth'
net = VSNet_realtime(model, True)

#save txt setting
vel_save=[]
ee_pos_next = [0,0,0]
gripper_command = [1,1]

bridge = CvBridge()
rgb_img = Image_PIL.open("/home/zls/Desktop/2000.png")

# the callback function of subscriber of realsense camera
def realsense_img_callback(data):
    global rgb_img
    img_array = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
    # rgb_img = Image_PIL.fromarray(img_array,"RGB")
    rgb_img = Image_PIL.fromarray(cv2.cvtColor(img_array,cv2.COLOR_BGR2RGB),"RGB")
    # rgb_img.save("/home/zls/Desktop/img.png")

# the callback function of subscriber of usb camera
def usb_img_callback(data):
    global rgb_img
    img_array = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
    rgb_img = Image_PIL.fromarray(img_array,"RGB")
    # rgb_img = Image_PIL.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB), "RGB")
    # rgb_img.save("/home/zls/Desktop/img.png")


def pub_pose_to_franka():
    # ros stuff
    pub_franka = rospy.Publisher('chatter_to_franka', Float32MultiArray, queue_size=10)

    rospy.init_node('publish_pose', anonymous=True)
    #realsense
    # rospy.Subscriber('image', Image, realsense_img_callback)
    #usb_cam
    rospy.Subscriber('/usb_cam/image_raw', Image, usb_img_callback)

    rate = rospy.Rate(30)  # 20hz
    next_comd = Float32MultiArray()
    while not rospy.is_shutdown():
        # print("Hey! I am publishing control command to franka!")
        start_time = time.time()
        vel = img2control(rgb_img)

        # de normalialize the control signal
        vel[0] = vel[0] * (label_0_max - label_0_min) + label_0_min
        vel[1] = vel[1] * (label_1_max - label_1_min) + label_1_min
        vel[2] = vel[2] * (label_2_max - label_2_min) + label_2_min

        next_comd.data = [vel[0], vel[1], vel[2],vel[3]]

        pub_franka.publish(next_comd)
        rate.sleep()
        end_time = time.time()
        # print("time",end_time-start_time)



if __name__ == '__main__':
    try:
        pub_pose_to_franka()
    except rospy.ROSInterruptException:
        pass
