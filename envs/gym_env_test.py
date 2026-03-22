from gym_env import PickPlaceEnv
import numpy as np

# Initialize environment
env = PickPlaceEnv(gui=True)
obs, info = env.reset()

# Tracking total reward for the episode
episode_reward = 0

for i in range(2000):
    # Sample random action: [dx, dy, dz, grasp]
    action = env.action_space.sample()
    
    # Gymnasium step returns 5 values
    obs, reward, terminated, truncated, info = env.step(action)

    episode_reward += reward

    # Print progress every 10 steps to keep the console readable
    if i % 10 == 0:
        print(f"Step: {i} | Current Reward: {reward:.4f} | Total Episode Reward: {episode_reward:.2f}")

    # Check if the task is done (cube lifted) or failed (time limit)
    if terminated or truncated:
        print("--- Episode Finished! Resetting... ---")
        obs, info = env.reset()
        episode_reward = 0

env.close()