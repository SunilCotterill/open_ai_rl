import gym
from stable_baselines3 import PPO
import os
from snakeenv import SnakeEnv

TIMESTEPS = 10000

models_dir = "models"
log_dir ="logs"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


env = SnakeEnv()
env.reset()

# MLP = Multi Layer Perceptron
# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log= log_dir)
model= PPO.load("models/450000.zip", env, tensorboard_log= log_dir)

for i in range(1,1000000):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*i + 450000}")

env.close()




