#!/usr/bin/env python
# license removed for brevity
import rospy
import sys
sys.path.append('../src')
import numpy as np
from autolab_core import RigidTransform
from std_msgs.msg import Float32MultiArray
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import JointPositionSensorMessage, ShouldTerminateSensorMessage
from franka_interface_msgs.msg import SensorDataGroup
from franka_robot import FrankaRobot


dh_params = np.array([[0, 0.333, 0, 0],
                      [0, 0, -np.pi / 2, 0],
                      [0, 0.316, np.pi / 2, 0],
                      [0.0825, 0, np.pi / 2, 0],
                      [-0.0825, 0.384, -np.pi / 2, 0],
                      [0, 0, np.pi / 2, 0],
                      [0.088, 0, np.pi / 2, 0],
                      [0, 0.107, 0, 0],
                      [0, 0.1034, 0, 0]])

fr = FrankaRobot('/home/zls/king/doing/visualtocontrol/Kinematics/franka_robot.urdf', dh_params, 7)

fa = FrankaArm()
grasping_scale = 0.06

ee_current = RigidTransform(rotation=np.array([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, -1],
]), translation=np.array([0.3069, 0, 0.4867]),
    from_frame='franka_tool', to_frame='world')
ee_pos_next = RigidTransform(rotation=np.array([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, -1],
]), translation=np.array([0.3069, 0, 0.4867]),
    from_frame='franka_tool', to_frame='world')

# initialize the control signal
command_signal = [0,0,0,0]


def franka_callback(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard it.")
    rospy.loginfo("received data: %s", data.data)
    global command_signal
    command_signal = data.data


output =[]
cartesian_pos=[]

def toFranka():
    # home pose and next pose initialization and definition
    # HOME_POSE = RigidTransform(rotation=np.array([
    #     [1, 0, 0],
    #     [0, -1, 0],
    #     [0, 0, -1],
    # ]), translation=np.array([0.4, 0, 0.30]),
    #     from_frame='franka_tool', to_frame='world')

    fa.reset_joints()
    # non-up-down grasping
    # HOME_POSE_non = RigidTransform(rotation=np.array([
    #     [1, 0, 0],
    #     [0, -1, 0],
    #     [0, 0, -1],
    # ]), translation=np.array([0.5, 0, 0.80]),
    #     from_frame='franka_tool', to_frame='world')
    #
    # HOME_POSE_non = RigidTransform(rotation=np.array([
    #     [1, 0, 0],
    #     [0, -1, 0],
    #     [0, 0, -1],
    # ]), translation=np.array([0.5, 0, 0.7]),
    #     from_frame='franka_tool', to_frame='world')
    #
    #
    # rotation_delta = RigidTransform(
    #     rotation=RigidTransform.y_axis_rotation(np.deg2rad(45)),
    #     from_frame=HOME_POSE_non.from_frame,to_frame=HOME_POSE_non.from_frame
    # )
    # HOME_POSE_non = HOME_POSE_non*rotation_delta
    # fa.goto_pose(HOME_POSE_non,use_impedance=False)
    #
    # # for non up-down grasping
    # delta = RigidTransform(translation=np.array([0.3, 0, -0.3]),
    #                        rotation=RigidTransform.y_axis_rotation(np.deg2rad(90)),
    #     from_frame='franka_tool', to_frame='franka_tool')

    #for the up-down
    delta = RigidTransform(translation=np.array([0.1, 0, 0]),
        from_frame='franka_tool', to_frame='franka_tool')

    ee_init = fa.get_pose()
    HOME_POSE_up_down = ee_init.copy()
    HOME_POSE_up_down = ee_init*delta
    fa.goto_pose(HOME_POSE_up_down)
    rospy.loginfo("wait for franka_callback")
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    # rospy.init_node('robot_info_listener', anonymous=True)

    rospy.Subscriber("chatter_to_franka", Float32MultiArray, franka_callback)
    # rospy.Subscriber("sub_gripper", Float32, gripper_callback)
    # Rate = rospy.Rate(20)
    # Rate.sleep()
    # spin() simply keeps python from exiting until this node is stopped

    rospy.loginfo('Initializing Sensor Publisher')
    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
    rate = rospy.Rate(30)


    rospy.loginfo('Publishing pose trajectory...')

    joint_now = fa.get_joints()
    ee_now, H = fr.ee(joint_now)

    ee_now_disired_rigid = RigidTransform(
        translation=np.array(ee_now[0:3]),
        rotation=H[0:3, 0:3],
        from_frame="ee", to_frame="world")

    # ee_next_disired_rigid = RigidTransform(
    #     translation=np.array(ee_now[0:3]),
    #     rotation=H[0:3, 0:3],
    #     from_frame="ee", to_frame="world")

    # To ensure skill doesn't end before completing trajectory, make the buffer time much longer than needed
    fa.goto_joints(joint_now, duration=5, dynamic=True, buffer_time=20,ignore_virtual_walls=True)

    fa.open_gripper()
    i = 0
    dt_x, dt_y, dt_z = 0.05, 0.05, 0.05
    # dt_x, dt_y, dt_z = 0.08, 0.08, 0.05

    has_closed = False
    is_grasped = False
    current_gripper_width = 0.8
    reach_flag = 0
    init_time = rospy.Time.now().to_time()

    while not rospy.is_shutdown():

        timestamp = rospy.Time.now().to_time() - init_time

        vel = [command_signal[0], command_signal[1], command_signal[2]]
        #note use different camera will have different transformation, this is because of the install pose of different camera.
# for the realsense cam
#         T_delta = RigidTransform(
#             translation=np.array([vel[1]*dt_x, vel[0]*dt_y, (-vel[2]*dt_z)/2]),
#             from_frame="ee", to_frame="ee")
        # for the usb cam
        T_delta = RigidTransform(
            translation=np.array([-vel[1]*dt_x, -vel[0]*dt_y, (-vel[2]*dt_z)/2]),
            from_frame="ee", to_frame="ee")
        print([vel[1]*dt_x, vel[0]*dt_y, -vel[2]*dt_z],"Pos_add")
        #save
        output.append((vel[1],vel[0],-vel[2]/2,command_signal[3]))
        np.savetxt("/home/zls/Desktop/output.txt", output)

        ee_next_disired_rigid = ee_now_disired_rigid * T_delta
        cartesian_pos.append(ee_next_disired_rigid.translation)
        np.savetxt("/home/zls/Desktop/cartesian_pos.txt", cartesian_pos)


        #rigidform to [x,y,z,r,p,y]
        ee_next_desired = fr.decompose_homogeneous_matrx(ee_next_disired_rigid.matrix)

        # get the current joint of franka
        joint_now = fa.get_joints()
        #IK
        joint_next = fr.inverse_kinematics(ee_next_desired,joint_now)

        traj_gen_proto_msg = JointPositionSensorMessage(
            id=i, timestamp=rospy.Time.now().to_time() - init_time,
            joints=joint_next
        )
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.JOINT_POSITION)
        )


        if not has_closed:
            current_gripper_width = command_signal[3]*0.06+0.04
            if current_gripper_width>0.08:
                current_gripper_width = 0.08
            if current_gripper_width<=0:
                current_gripper_width = 0
        else:
            current_gripper_width = 0.08

        if fa.get_gripper_is_grasped():
            print("I grasped the object!")
            has_closed = True
            fa.goto_pose(HOME_POSE_up_down)
        # use realsense camera
        # fa.goto_gripper(current_gripper_width, speed=1.0,block=False,force=0.02)
        # use usb camera
        fa.goto_gripper(0.08, speed=1.0,block=False,force=0.02)
        if abs(vel[0]*0.05)<0.001 and abs(vel[1]*0.05)<0.001 and abs(vel[2]*0.05/2)<0.0002 and vel[0]!=0:
            fa.stop_skill()
            reach_flag=1
        if reach_flag==1:
            fa.goto_gripper(0.015,force=0.005)
            now_ee = fa.get_pose()
            ee_destination = now_ee.copy()
            detlat_destination =RigidTransform(translation=np.array([0,0,-0.1]),to_frame='franka_tool',from_frame='franka_tool')
            ee_destination = now_ee*detlat_destination
            fa.goto_pose(ee_destination)
            # fa.reset_joints()

        rospy.loginfo('Publishing: ID {}'.format(traj_gen_proto_msg.id))

        pub.publish(ros_msg)
        # update
        ee_now_disired_rigid = ee_next_disired_rigid.copy()
        ee_now_disired_rigid.rotation = H[0:3, 0:3]
        i+=1
        rate.sleep()

    # Stop the skill
    # Alternatively can call fa.stop_skill()
    term_proto_msg = ShouldTerminateSensorMessage(timestamp=rospy.Time.now().to_time() - init_time,
                                                  should_terminate=True)
    ros_msg = make_sensor_group_msg(
        termination_handler_sensor_msg=sensor_proto2ros_msg(
            term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
    )
    pub.publish(ros_msg)

    rospy.loginfo('Done')


if __name__ == '__main__':
    toFranka()
