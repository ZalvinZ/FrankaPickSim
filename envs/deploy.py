from stable_baselines3 import PPO
from gym_env import PickPlaceEnv
import time

# Load env with GUI ON
env = PickPlaceEnv(gui=True)

# Load trained model
model = PPO.load("ppo_pickplace")

obs, _ = env.reset()

while True:
    # Predict best action
    action, _ = model.predict(obs, deterministic=True)

    # Step env
    obs, reward, terminated, truncated, info = env.step(action)

    time.sleep(1/60)

    if terminated or truncated:
        obs, _ = env.reset()
