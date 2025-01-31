import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from pysc2.env import sc2_env
from pysc2.lib import actions
from absl import flags

# Ensure SC2 path is correctly set
os.environ["SC2PATH"] = r"D:\StarCraft II"

# Initialize absl flags
FLAGS = flags.FLAGS
FLAGS([''])

# Define SC2 Gym Environment
class SC2GymEnv(gym.Env):
    def __init__(self, map_name="Simple64"):
        super(SC2GymEnv, self).__init__()

        self.env = sc2_env.SC2Env(
            map_name=map_name,
            players=[
                sc2_env.Agent(sc2_env.Race.terran),
                sc2_env.Bot(sc2_env.Race.zerg, sc2_env.Difficulty.medium)
            ],
            agent_interface_format=sc2_env.parse_agent_interface_format(
                feature_screen=84,
                feature_minimap=64,
                use_feature_units=True
            ),
            step_mul=2,
            game_steps_per_episode=0,
            visualize=True  # Enable visualization to watch AI play
        )

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=float)
        self.action_space = gym.spaces.Discrete(len(actions.FUNCTIONS))

    def step(self, action):
        """Execute an action in the SC2 environment."""
        try:
            obs, reward, done, info = self.env.step([actions.FunctionCall(action, [])])
            return obs[0].observation["feature_screen"], reward, done, info
        except Exception as e:
            print(f"Error in step execution: {e}")
            return self.reset(), 0, True, {}

    def reset(self):
        """Reset the SC2 environment."""
        obs = self.env.reset()
        return obs[0].observation["feature_screen"]

# Load trained model
model = PPO.load("sc2_ppo_model")

# Create RL environment
env = make_vec_env(lambda: SC2GymEnv(), n_envs=1)

# Run the trained AI
obs = env.reset()
done = False  # Define 'done' to track episode end

while not done:  # Keep running until the episode ends
    action, _ = model.predict(obs)  # AI chooses action
    obs, reward, done, _ = env.step(action)  # Step environment forward
    print(f"Action: {action}, Reward: {reward}, Done: {done}")
