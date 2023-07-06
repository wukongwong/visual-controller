# Visual to control
To ahieve a dynamic visual servo controller for a wide range of objects.

# Requirements:(We test our project in Ubuntu 20.04)
1. We use conda to manage the python env, install the environment.yaml;
2. Install pybullet_object according to https://github.com/eleramp/pybullet-object-models;
3. For the real experiments, you need to install frankapy and franka-interface accrding to https://iamlab-cmu.github.io/frankapy/index.html and https://github.com/iamlab-cmu/franka-interface.

# File struture introduction:
1. models and src file contains robot simulation model;
2. trainNN file contains scripts run the NN(controller), run the train.py, which need to modify the dataset direction based on yours;
3. scripts file contain code running the simulator environment, and real experments;

--run_camera.py can lauch the realsense camera and publish image to rostopic;

--run_model.py can run the trained model and publish output to rostopic;

--roRobot_joint.py can accept the output from run_model.py and use frankapy to control the real robot.

--dynamic_visual_servo.py is the srcripts that launch a pybullet simulator that verify the simulation performance. Pls note we do not impement the gripper part, but the output can be recored if need；

4. runNN file contains code file when doing the NN inference;
5. yolov5 contains the code that detects the bouding box of target objects;
6. kinematics file contains the frankpy kinematics, such as IK.

# dataset and trained model:
They can be found at https://drive.google.com/drive/folders/1lPT0FiArjju4aQTpktE_4pZ7mRmUr6sX?usp=sharing
# How to use:

1. For simualtion: just run dynamic-visual_servo.py;

2. For real world: In our case, we have two PCs, one is use to control the franka, on this pc, we lanuch the run_camera.py and frakapy;
On annother PC with 1080Ti GPU, we run run_model.py and toRobot_joint.py that run the NN and sendout the control signals. We use ros to do the communication between the two PCs.

# Things needs to be noted:
1. To increase the inference speed in the real world, for the goal image branch, we load the goal iamge NN branch output since the goal image is fixed;
2. In the real world, we decrease the velocity to half in Z direction considering the real-time issue. 

# All rights are reserved to the author 
Commercial use is prohibited. Dop an email to wukongwoong@gmail.com for permission if you want to use it for academic purpose.

