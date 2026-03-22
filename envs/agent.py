from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from gym_env import PickPlaceEnv
import time
import torch
import os


#create directories for models and logs
device = "cuda" if torch.cuda.is_available() else "cpu"
models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

#making sure the directories exist
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

#Create env (NO GUI during training)
env = PickPlaceEnv(gui=False)


#Check env correctness
check_env(env, warn=True)

#Create PPO agent
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99
)

# To continue training from a checkpoint, use the line below instead
# model = PPO.load("./models/1760167333/1400000.zip", env=env, verbose=1, tensorboard_log=logdir, device=device)

#Train
model.learn(total_timesteps=200_000)

# Save trained model
model.save("ppo_pickplace")

print("Training complete, model saved!")
