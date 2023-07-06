import os
import sys
sys.path.append('../src')
from PIL import Image
import pybullet as p
import time
import math  as m
from panda_ball import Panda
from scipy.interpolate import CubicSpline
import numpy as np
from inference import VSNet
from pybullet_object_models import ycb_objects


#Calculating the outer tangent circle of a mask
def calculate_outer_tangent_circle(seg):
    # seg.save("/home/wukong/Desktop/seg.png")
    [h, w] = np.shape(seg)
    print(h,w)
    #Calculate the center of the mask
    center_x = 0
    center_y = 0
    center_z = 0
    for i in range(h):
        for j in range(w):
            if seg.getpixel((i,j)) != 0:
                center_x += i
                center_y += j
                center_z += 1
    center_x = center_x/center_z
    center_y = center_y/center_z
    center_z = center_z/center_z
    print(center_x,center_y,center_z)
    #Calculate the radius of the mask
    radius = 0
    for i in range(h):
        for j in range(w):
            if seg.getpixel((i,j)) != 0 and (i-center_x)**2+(j-center_y)**2 > radius**2:
                radius = (i-center_x)**2+(j-center_y)**2
    radius = radius/center_z
    radius = radius**0.5

    print(radius,'radius')
    #Determining whether a pixel point is inside a circle
    for i in range(h):
        for j in range(w):
            if ((i-center_x)**2+(j-center_y)**2)**0.5 > radius:
                seg.putpixel((i,j),0)
            else:
                seg.putpixel((i,j),255)
    _,mask_ratio = caculate_mask_area(seg)
    if mask_ratio >0.12:
        radius= radius-50
        for i in range(h):
            for j in range(w):
                if ((i - center_x) ** 2 + (j - center_y) ** 2) ** 0.5 > radius:
                    seg.putpixel((i, j), 0)
                else:
                    seg.putpixel((i, j), 255)
    return seg

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
    print(mask_ratio,'mask_ratio')
    return area,mask_ratio

# N = 50#N is the total number of points in circle.
h_start = 0.4
h_end = 0.15

#n is the nth point of the circle.
def demo_taj(r,n,N):

    #cacluate the initial point
    theta = ((m.pi*2)/N)*n
    start_point = [r*m.sin(theta)+0.5, r*m.cos(theta),h_start]
    print(start_point)
    #caculate the end point
    end_point = [0.5,0,h_end]

    return start_point,end_point

def taj_planning(start_point,end_point):
    #(0.3,0.2,0.5) to (0.3,0.2,0.15)
    t_x=[0, 2]
    t_y=[0, 2]
    t_z=[0, 2]
    f_x = CubicSpline(t_x, [start_point[0],end_point[0]],bc_type='clamped')
    f_y = CubicSpline(t_y, [start_point[1],end_point[1]],bc_type='clamped')
    f_z = CubicSpline(t_z, [start_point[2],end_point[2]],bc_type='clamped')

    return f_x, f_y, f_z

model_dir = '/media/wukong/My Passport/save/VSNet-train/'
model = model_dir + 'model.pth'

bottleneck_value = 0.8
duration = 3000
stepsize = 1e-3

robot = Panda(stepsize)
robot.setControlMode("position")
#YCB objects
# obj_names = ['YcbBanana','YcbChipsCan','YcbCrackerBox','YcbFoamBrick','YcbGelatinBox','YcbHammer','YcbMasterChefCan'
#              ,'YcbMediumClamp','YcbMustardBottle','YcbPear','YcbPottedMeatCan','YcbPowerDrill','YcbScissors','YcbStrawberry','YcbTennisBall','YcbTomatoSoupCan']
obj_names = ['YcbPear']
#random pick one object
obj_name = obj_names[np.random.randint(0,len(obj_names))]
print(obj_name,'obj_name')
# flags = p.URDF_USE_INERTIA_FROM_FILE

obj_id = p.loadURDF(os.path.join(ycb_objects.getDataPath(), obj_name, "model.urdf"), useFixedBase=True)
# obj_id = p.loadURDF("ball/ball.urdf",useFixedBase=True)

p.changeDynamics(obj_id,-1,restitution=.95, linearDamping = 1e-2, angularDamping = 1e-2)
# default_pos = [0.5,0.1,0.0]
# default_ori = [1,0,0,0]

#define the start and end point
start_point, end_point = demo_taj(0.0, 0,50)
fx, fy, fz = taj_planning(start_point, end_point)


#inite the robot arm and the object
planning_point_0 = [fx(float(0/1000.00)),fy(float(0/1000.00)),fz(float(0/1000.00))]
print("planning_point_0",planning_point_0)

#cartesion to joint by IK
# target_pos = robot.solveInverseKinematics(planning_point_0,[0.924,0,0.383,0])
target_pos = robot.solveInverseKinematics(planning_point_0,[1,0,0,0])
error = 1
while error>0.001:
    # target_pos = robot.solveInverseKinematics(planning_point_0,[0.924,0,0.383,0])
    target_pos = robot.solveInverseKinematics(planning_point_0,[1,0,0,0])

    robot.setTargetPositions(target_pos)
    ee_pos = robot.getLinkState()[0]
    error = m.sqrt((ee_pos[0]- planning_point_0[0])*(ee_pos[0]- planning_point_0[0])+
    (ee_pos[1]- planning_point_0[1])*(ee_pos[1]- planning_point_0[1])+
    (ee_pos[2]- planning_point_0[2])*(ee_pos[2]- planning_point_0[2]))
    print("error",error)
    print (ee_pos)
    robot.step()

ee_pos = robot.getLinkState()[0]
ee_ori = robot.getLinkState()[1]
# transfer the qua to a 3*3 matrix
ee_ori_matrix = p.getMatrixFromQuaternion(ee_ori)
ee_ori_matrix = np.array(ee_ori_matrix)
ee_ori_matrix = ee_ori_matrix.reshape(3,3)

# define the ee transformation matrix
ee_transformation = np.zeros((4,4))
ee_transformation[0:3,0:3] = ee_ori_matrix
ee_transformation[0:3,3] = ee_pos
ee_transformation[3,3] = 1

#define the translation matrix
T = np.array([[1,0,0.0,0.0],[0,1,0.0,0.0],[0,0,1,0.4],[0,0,0.0,1]])
T_output = np.array([[1,0,0.0,0.0],[0,1,0.0,0.0],[0,0,1,0.0],[0,0,0.0,1]])

#define the rotation matrix
R = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
#multiply the translation matrix with ee_pos
obj_pos = np.dot(ee_transformation,T)
#get the object position
default_pos = obj_pos[0:3,3]
default_ori = [1,0,0,0]
p.resetBasePositionAndOrientation(obj_id, default_pos, default_ori)

#save txt setting
pos_point=[]
vel_save=[]
far = 3.1
near = 0.01
ee_pos_next = [0,0,0]
#run the robot simulation
for i in range(int(duration/stepsize)):

    if i%1000 == 0:
        print("Simulation time: {:.3f}".format(robot.t))

    if i%20 == 0:
        _, _, segImg = robot.getCameraImage()
        segim = segImg * 1. / 255.
        seg = Image.fromarray(np.uint8(segImg * 255))

        #velocity
        img_current = calculate_outer_tangent_circle(seg)

        img_goal = '../goal.png'
        # net = VSNet_realtime(model,False)
        net = VSNet(model,False)

        output = net.infer(img_current, img_goal)
        print(output)

        #output progress
        _, mask_ratio = caculate_mask_area(seg)

        if mask_ratio>bottleneck_value:
            vel =[0,0,0]
        else:
            vel = [output[0],output[1],output[2]]
        #caculate the current pos
        ee_pos_current = robot.getLinkState()[0]
        ee_ori_current = robot.getLinkState()[1]
        # transfer the qua to a 3*3 matrix
        ee_ori_matrix_c = p.getMatrixFromQuaternion(ee_ori_current)
        ee_ori_matrix_c = np.array(ee_ori_matrix_c)
        ee_ori_matrix_c = ee_ori_matrix_c.reshape(3, 3)

        # define the ee transformation matrix
        ee_transformation = np.zeros((4, 4))
        # ee_transformation[0:3, 0:3] = ee_ori_matrix_c
        ee_transformation[0:3, 0:3] = np.eye(3)

        ee_transformation[0:3, 3] = ee_pos_current
        ee_transformation[3, 3] = 1
        # calculate the transfromation matrix from output
        # T_output[0:3,3] = [vel[0],-vel[1],-vel[2]]#*1.0
        T_output[0:3,3] = [vel[0],vel[1],vel[2]]#*1.0

        ee_pos_next_trans = np.dot(ee_transformation,T_output)
        ee_pos_next = ee_pos_next_trans[0:3,3]
        # ee_pos_next_test = [ee_pos_current[0] + output[0],ee_pos_current[1] + output[1],ee_pos_current[2] + output[2]]

    if i<=4000:
        planning_point = [fx(float(i/1000.00)),fy(float(i/1000.00)),fz(float(i/1000.00))]
        pos_point.append(planning_point)

    ee_now = ee_pos_next
    # print("ee_now",ee_now)
    #cartesion to joint by IK
    # target_pos = robot.solveInverseKinematics(ee_now,[0.924,0,0.383,0])
    target_pos = robot.solveInverseKinematics(ee_now,[1,0,0,0])

    robot.setTargetPositions(target_pos)
    robot.step()
    time.sleep(robot.stepsize)
    if i>2000:
        break
