import pybullet as p
import pybullet_data
import numpy as np
import time

class PickPlaceEnv:
    def __init__(self, gui=True):
        self.gui = gui
        self.cid = p.connect(p.GUI if gui else p.DIRECT) #Check to see if we using GUI otherwise sets to DIRECT.
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #Calls additional libraries such as physics, texture 3D models
        p.setGravity(0, 0, -9.81) #Setting gravity conditions in x,z,y format

        self._load_world()
        self._load_robot()

        #Variabl heights
        TABLE_Z = 0.62
        CUBE_Z = TABLE_Z + 0.045


    def _load_world(self):
        self.plane = p.loadURDF("plane.urdf") #Loads in plain urdf
        self.table = p.loadURDF(
            "table/table.urdf",
            basePosition=[0.5, 0, 0]
        )#imports a table from pybullet. Sets the base position. (x,z,y)

        self.object = p.loadURDF(
            "cube_small.urdf",
            basePosition=[0.6, 0, 0.67]
        ) #Loads in a small cube from pybullet and places it on top of the table. (x,z,y)
    
    def reset_object(self):
        x = np.random.uniform(0.45, 0.67)
        y = np.random.uniform(-0.2, 0.2)
        p.resetBasePositionAndOrientation(
            self.object,
            [x, y, 0.65],
            [0, 0, 0, 1]
        )

    def _load_robot(self):
        self.robot = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0, 0, 0.62],  # <-- FIX
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
        )#Imports a franka panda robot and places it on top of the table

        self.ee_link = 11 #0 panda_link1, panda_link2, panda_link3, panda_link4, panda_link5, panda_link6, panda_link7, panda_link8, panda_hand, panda_leftfinger, panda_rightfinger, panda_grasptarget

    def step_sim(self, steps=240):
        for _ in range(steps):
            p.stepSimulation()
            if self.gui:
                time.sleep(1 / 240)
    
    def get_ee_pose(self):
        state = p.getLinkState(self.robot, self.ee_link)
        return np.array(state[0]), np.array(state[1])

    def move_ee(self, dx, dy, dz):
        pos, orn = self.get_ee_pose()
        target = pos + np.array([dx, dy, dz])

        joint_poses = p.calculateInverseKinematics(
            self.robot,
            self.ee_link,
            target,
            orn
        )
        # for i in range(p.getNumJoints(self.robot)):
        #     info = p.getJointInfo(self.robot, i)
        #     print(i, info[12].decode())

        for j in range(7):
            p.setJointMotorControl2(
                self.robot,
                j,
                p.POSITION_CONTROL,
                joint_poses[j],
                force=200
            )
    
    def get_camera_image(self):
        view = p.computeViewMatrix(
            cameraEyePosition=[0.8, 0, 1.2],
            cameraTargetPosition=[0.5, 0, 0.6],
            cameraUpVector=[0, 0, 1]
        )

        proj = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=1.0,
            nearVal=0.1,
            farVal=3.0
        )

        width, height, rgb, _, _ = p.getCameraImage(
            64, 64, view, proj
        )

        rgb = np.reshape(rgb, (64, 64, 4))[:, :, :3]
        return rgb.astype(np.uint8)



#TODO: Set up Depth camera