import math
import time

import numpy as np
from franka_robot import FrankaRobot

if __name__ == '__main__':
	dh_params = np.array([[0, 0.333, 0, 0],
						  [0, 0, -math.pi/2, 0],
						  [0, 0.316, math.pi/2, 0],
						  [0.0825, 0, math.pi/2, 0],
						  [-0.0825, 0.384, -math.pi/2, 0],
						  [0, 0, math.pi/2, 0],
						  [0.088, 0, math.pi/2, 0],
						  [0, 0.107, 0, 0],
						  [0, 0.1034, 0, 0]])
	fr = FrankaRobot('franka_robot.urdf', dh_params, 7)
	joints = [0, -math.pi/4, 0, -3*math.pi/4, 0, math.pi/2, math.pi/4]
	print(joints)
	ee = fr.ee(joints)
	ee[0][1] += 0.01
	time_start = time.time()
	new_joints = fr.inverse_kinematics(ee, joints)
	time_end = time.time()
	print(time_end-time_start,"time")
	new_ee = fr.ee(new_joints)
	print(new_joints)

	print(np.allclose(new_ee,ee))