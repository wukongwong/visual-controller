import pybullet as p

class Panda:
    def __init__(self, stepsize=1e-3, realtime=0):
        self.t = 0.0
        self.stepsize = stepsize
        self.realtime = realtime

        self.control_mode = "torque"

        self.position_control_gain_p = [0.01,0.01,0.01,0.01,0.01,0.01,0.01]
        self.position_control_gain_d = [1.0,1.0,1.0,1.0,1.0,1.0,1.0]
        self.max_torque = [100,100,100,100,100,100,100]

        # connect pybullet
        self.physicsClient = p.connect(p.GUI)
        # self.physicsClient = p.connect(p.DIRECT)

        # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=30, cameraPitch=-20, cameraTargetPosition=[0, 0, 0.5])

        p.resetSimulation()
        p.setTimeStep(self.stepsize)

        p.setRealTimeSimulation(self.realtime)
        p.setGravity(0,0,-9.81)

        # load models
        p.setAdditionalSearchPath("../models")

        self.plane = p.loadURDF("plane/plane.urdf",
                                useFixedBase=True)
        p.changeDynamics(self.plane,-1,restitution=.95)


        # ball
        # self.ball = p.loadURDF("ball/ball.urdf",useFixedBase=True)
        # p.changeDynamics(self.ball,-1,restitution=.95, linearDamping = 1e-2, angularDamping = 1e-2)
        # self.default_ball_pos = [0.5,0.0,0.05]
        # self.default_ball_ori = [0,0,0,1]
        # p.resetBasePositionAndOrientation(self.ball, self.default_ball_pos, self.default_ball_ori)

        # #YCB objects
        # flags = p.URDF_USE_INERTIA_FROM_FILE
        # obj_id = p.loadURDF(os.path.join(ycb_objects.getDataPath(), 'YcbBanana', "model.urdf"), [1., 0.0, 0.8],
        #                     flags=flags)
        # robot
        self.robot = p.loadURDF("panda/panda.urdf",
                                useFixedBase=True,
                                flags=p.URDF_USE_SELF_COLLISION)
        # self.robot = p.loadURDF("panda/panda_gripper.urdf",
        #                         useFixedBase=True,
        #                         flags=p.URDF_USE_SELF_COLLISION)


        # robot parameters
        self.dof = p.getNumJoints(self.robot) - 1 # Virtual fixed joint between the flange and last link
        print(self.dof)
        if self.dof != 7:
            raise Exception('wrong urdf file: number of joints is not 7')

        self.joints = []
        self.q_min = []
        self.q_max = []
        self.target_pos = []
        self.target_torque = []

        for j in range(self.dof):
            joint_info = p.getJointInfo(self.robot, j)
            self.joints.append(j)
            self.q_min.append(joint_info[8])
            self.q_max.append(joint_info[9])
            self.target_pos.append((self.q_min[j] + self.q_max[j])/2.0)
            self.target_torque.append(0.)

        self.reset()

    def reset(self):
        self.t = 0.0
        self.control_mode = "torque"
        for j in range(self.dof):
            self.target_pos[j] = (self.q_min[j] + self.q_max[j])/2.0
            self.target_torque[j] = 0.
            p.resetJointState(self.robot,j,targetValue=self.target_pos[j])

        self.resetController()

    def step(self):
        self.t += self.stepsize
        p.stepSimulation()

    # robot functions
    def resetController(self):
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=self.joints,
                                    controlMode=p.VELOCITY_CONTROL,
                                    forces=[0. for i in range(self.dof)])

    def setControlMode(self, mode):
        if mode == "position":
            self.control_mode = "position"
        elif mode == "torque":
            if self.control_mode != "torque":
                self.resetController()
            self.control_mode = "torque"
        else:
            raise Exception('wrong control mode')

    def setTargetPositions(self, target_pos):
        self.target_pos = target_pos
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=self.joints,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=self.target_pos,
                                    forces=self.max_torque,
                                    positionGains=self.position_control_gain_p,
                                    velocityGains=self.position_control_gain_d)

    def setTargetTorques(self, target_torque):
        self.target_torque = target_torque
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=self.joints,
                                    controlMode=p.TORQUE_CONTROL,
                                    forces=self.target_torque)

    def getJointStates(self):
        joint_states = p.getJointStates(self.robot, self.joints)
        joint_pos = [x[0] for x in joint_states]
        joint_vel = [x[1] for x in joint_states]
        return joint_pos, joint_vel

    def getLinkState(self):
        jj = p.getJointInfo(self.robot,7)
        ee_state = p.getLinkState(self.robot, 7)
        # print(p.getNumJoints(self.robot))
        #
        # print(p.getJointInfo(self.robot,7))
        # print(ee_state)
        return ee_state

    def solveInverseDynamics(self, pos, vel, acc):
        return list(p.calculateInverseDynamics(self.robot, pos, vel, acc))

    def solveInverseKinematics(self, pos, ori):
        return list(p.calculateInverseKinematics(self.robot, 7, pos, ori))

    # ball funtions
    def resetBall(self, ball_pos=None, ball_ori=None):
        if ball_pos is None:
            ball_pos = self.default_ball_pos
        if ball_ori is None:
            ball_ori = self.default_ball_ori
        p.resetBasePositionAndOrientation(self.ball, ball_pos, ball_ori)

    def getBallStates(self):
        ball_states = p.getBasePositionAndOrientation(self.ball)
        ball_pos = list(ball_states[0])
        ball_ori = list(ball_states[1])
        return ball_pos, ball_ori


    def getCameraImage(self,physicsClient=None):
        """
        INTERNAL METHOD, Computes the OpenGL virtual camera image. The
        resolution and the projection matrix have to be computed before calling
        this method, or it will crash

        Returns:
            camera_image - The camera image of the OpenGL virtual camera
        """


        ee_state = self.getLinkState()
        ee_pos = ee_state[0]

        camera_pos = [ee_pos[0], ee_pos[1], ee_pos[2]]
        # camera_pos = [0.3, 0.2, ee_pos[2]]


        # calculate the transformation matrix from position and orientation
        # of the camera to the world frame
        camera_ori = ee_state[1]
        rotation = p.getMatrixFromQuaternion(camera_ori)
        up_vector = [rotation[1], -rotation[4], rotation[7]]
        z_vector = [rotation[2], rotation[5], rotation[8]]

        camera_target = [
            camera_pos[0] + z_vector[0] * 0.35,
            camera_pos[1] + z_vector[1] * 0.35,
            camera_pos[2] + z_vector[2] * 0.35]

        # print(up_vector,'upvector')
        # print([0,1,0],'upvector_origin')
        # print(camera_target,'camera_target')
        # print([ee_pos[0],ee_pos[1],self.default_ball_pos[2]],'camera_target_origin')
        view_matrix = p.computeViewMatrix(
            camera_pos,
            cameraTargetPosition=camera_target,
            cameraUpVector=up_vector,

        )

        # view_matrix = p.computeViewMatrix(
        #     camera_pos,
        #     cameraTargetPosition=[ee_pos[0],ee_pos[1],self.default_ball_pos[2]],
        #     cameraUpVector=[0, 1, 0],
        #
        # )
        projectionMatrix = p.computeProjectionMatrixFOV(
                            fov=60.0,
                            aspect=1.0,
                            nearVal=0.01,
                            farVal=3.1

        )


        width, height, rgbImg, depthImg, segImg = p.getCameraImage(
            480,
            480,
            view_matrix,
            projectionMatrix,
            shadow=False

        )
        return rgbImg,depthImg,segImg

if __name__ == "__main__":
    robot = Panda(realtime=1)
    while True:
        pass
