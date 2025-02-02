import gym
from stable_baselines3 import PPO
from sc2_gym_env import SC2GymEnv
import torch

env = SC2GymEnv()

# Use CNN policy because SC2 observations are images
model = PPO("CnnPolicy", env, verbose=1, device="cuda" if torch.cuda.is_available() else "cpu")

# Train AI for 500,000 steps
model.learn(total_timesteps=500000)

# Save trained model
model.save("sc2_full_ppo_agent")

# Close environment
env.close()
