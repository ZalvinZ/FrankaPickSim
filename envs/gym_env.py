import time
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces

class PickPlaceEnv(gym.Env):
    """
    A PyBullet environment for a Franka Panda robot to perform 
    pick-and-place tasks.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 240}

    def __init__(self, gui=True):
        super().__init__()

        # --- Connection Setup ---
        self.gui = gui
        if gui:
            self.cid = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        else:
            self.cid = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # --- Environment Attributes ---
        self.target_height = 0.8 
        self.ee_link = 11
        self.gripper_indices = [9, 10]
        self.max_steps = 250
        self.current_step = 0
        self.grasp_constraint = None

        # --- Spaces ---
        # Action: [dx, dy, dz, gripper_command]
        self.action_space = spaces.Box(
            low=np.array([-0.05, -0.05, -0.05, -1.0], dtype=np.float32),
            high=np.array([0.05, 0.05, 0.05, 1.0], dtype=np.float32),
        )

        # Observation: [ee_x, ee_y, ee_z, obj_x, obj_y, obj_z]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        # --- Initialize Assets ---
        self._load_world()
        self._load_robot()
        
        if gui:
            self._setup_camera()

    def _load_world(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self.plane = p.loadURDF("plane.urdf")
        self.table = p.loadURDF("table/table.urdf", basePosition=[0.5, 0, 0])
        self.object = p.loadURDF("cube_small.urdf", basePosition=[0.6, 0, 0.67])
        
        # Target marker sphere
        visual_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[0, 1, 0, 0.7])
        self.goal_marker = p.createMultiBody(
            baseVisualShapeIndex=visual_id, 
            basePosition=[0.6, 0, self.target_height]
        )

    def _load_robot(self):
        self.robot = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0, 0, 0.62],
            useFixedBase=True
        )

    def _setup_camera(self):
        p.resetDebugVisualizerCamera(
            cameraDistance=1.3,
            cameraYaw=35,
            cameraPitch=-30,
            cameraTargetPosition=[0.5, 0, 0.65]
        )

    def _get_obs(self):
        ee_pos = np.array(p.getLinkState(self.robot, self.ee_link)[0])
        cube_pos, _ = p.getBasePositionAndOrientation(self.object)
        return np.concatenate([ee_pos, cube_pos]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Reset Robot to Neutral
        neutral_positions = [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.7]
        for i, pos in enumerate(neutral_positions):
            p.resetJointState(self.robot, i, pos)

        # Randomize cube position on table
        x = np.random.uniform(0.5, 0.65)
        y = np.random.uniform(-0.15, 0.15)
        p.resetBasePositionAndOrientation(self.object, [x, y, 0.65], [0, 0, 0, 1])

        p.stepSimulation()
        return self._get_obs(), {}

    def _compute_reward(self, ee_pos, cube_pos):
        dist = np.linalg.norm(ee_pos - cube_pos)
        reward = -dist

        # Reward for being close to cube
        if dist < 0.05:
            reward += 2.0

        # Reward for lifting the object
        if cube_pos[2] > 0.65:
            reward += (cube_pos[2] - 0.65) * 200

        # High-altitude grasp reward
        if dist < 0.03 and cube_pos[2] > 0.68:
            reward += 5.0

        terminated = bool(cube_pos[2] > self.target_height)
        if terminated:
            reward += 100.0

        return float(reward), terminated

    def step(self, action):
        self.current_step += 1
        dx, dy, dz, gripper_cmd = action
        
        # 1. Update End Effector Position
        current_ee = np.array(p.getLinkState(self.robot, self.ee_link)[0])
        target_ee = current_ee + np.array([dx, dy, dz])
        
        # 2. Inverse Kinematics
        joint_poses = p.calculateInverseKinematics(self.robot, self.ee_link, target_ee)
        for i in range(7):
            p.setJointMotorControl2(
                self.robot, i, p.POSITION_CONTROL, joint_poses[i], force=200
            )

        # 3. Gripper Control
        gripper_pos = 0.04 if gripper_cmd > 0 else 0.00
        for i in self.gripper_indices:
            p.setJointMotorControl2(
                self.robot, i, p.POSITION_CONTROL, gripper_pos, force=100
            )

        # 4. Simulation Steps
        for _ in range(5):
            p.stepSimulation()
            if self.gui:
                time.sleep(1/480)

        # 5. Post-step Processing
        obs = self._get_obs()
        ee_pos, cube_pos = obs[:3], obs[3:]
        reward, terminated = self._compute_reward(ee_pos, cube_pos)
        truncated = bool(self.current_step >= self.max_steps)
        
        return obs, reward, terminated, truncated, {}

    def close(self):
        p.disconnect()

if __name__ == "__main__":
    env = PickPlaceEnv(gui=True)
    obs, info = env.reset()

    try:
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                obs, info = env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()