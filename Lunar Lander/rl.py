import gym
from stable_baselines3 import PPO, A2C
import os

def train_lunar(model_type, model_name, timestep = 10000):
    models_dir = "models/"+model_name
    log_dir ="logs"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


    env = gym.make("LunarLander-v2")
    env.reset()

    # MLP = Multi Layer Perceptron
    model = model_type("MlpPolicy", env, verbose=1, tensorboard_log= log_dir)

    for i in range(1,30):
        model.learn(total_timesteps=timestep, reset_num_timesteps=False, tb_log_name=model_name)
        model.save(f"{models_dir}/{timestep*i}")
    
    env.close()


# train_lunar(PPO, "PPO")
# train_lunar(A2C, "A2C")

env = gym.make("LunarLander-v2", render_mode="human")
model = A2C.load("models/A2C/210000.zip")
episodes = 10
for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        res = env.step(action)
        obs = res[0]
        reward = res[1]
        done = res[2]


env.close()

# episodes = 2


