import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time


class PickPlaceEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 240}

    def __init__(self, gui=True):
        super().__init__()

        # Connection
        self.gui = gui
        if gui:
            self.cid = p.connect(p.GUI)

            # Use a safer way to configure UI to avoid AttributeErrors
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        else:
            self.cid = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Define Attributes (Must be before loading assets)
        self.target_height = 0.8
        self.ee_link = 11
        self.gripper_indices = [9, 10]
        self.max_steps = 250
        self.current_step = 0
        self.grasp_constraint = None

        # Define Spaces
        self.action_space = spaces.Box(
            low=np.array([-0.05, -0.05, -0.05, -1.0], dtype=np.float32),
            high=np.array([0.05, 0.05, 0.05, 1.0], dtype=np.float32),
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32
        )

        # Load Assets
        self._load_world()
        self._load_robot()

        if gui:
            self._setup_camera()

    def _setup_camera(self):
        """Positions the camera to look at the table from a high angle."""
        p.resetDebugVisualizerCamera(
            cameraDistance=1.3,
            cameraYaw=35,
            cameraPitch=-30,
            cameraTargetPosition=[0.5, 0, 0.65]
        )

    def control_gripper(self, close=True):
        force = 50

        if close:
            target = 0.0  # closed
        else:
            target = 0.04  # open (adjust if needed)

        for joint in [9, 10]:
            p.setJointMotorControl2(
                bodyUniqueId=self.robot,
                jointIndex=joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target,
                force=force
            )

    def try_grasp(self):

        if self.grasp_constraint is not None:
            return

        ee_pos, _ = self.get_ee_pose()
        cube_pos, _ = p.getBasePositionAndOrientation(self.object)

        dist = np.linalg.norm(ee_pos - cube_pos)
