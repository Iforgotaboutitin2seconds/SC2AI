import os
import numpy as np
import gymnasium as gym
import torch
import msvcrt

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from absl import flags
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units

############################
# 1. Setup SC2 + ABSL FLAGS
############################
os.environ["SC2PATH"] = r"D:\StarCraft II"  # Adjust if needed
FLAGS = flags.FLAGS
FLAGS([""])  # Prevent absl.flags errors

###################################################################
# 2. Define a Custom SC2 Env with Extended Observations & Some Improved Action Handling
###################################################################
class SC2GymEnv(gym.Env):
    def __init__(self, map_name="Simple64"):
        super().__init__()
        self.map_name = map_name

        # Create the underlying SC2 environment
        self.env = sc2_env.SC2Env(
            map_name=map_name,
            players=[
                sc2_env.Agent(sc2_env.Race.terran),
                sc2_env.Bot(sc2_env.Race.zerg, sc2_env.Difficulty.medium),
            ],
            agent_interface_format=sc2_env.parse_agent_interface_format(
                feature_screen=84,
                feature_minimap=64,
                use_feature_units=True,
            ),
            step_mul=8,
            game_steps_per_episode=0,
            visualize=False,
        )

        # Observation space: (6,84,84)
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(6, 84, 84),
            dtype=np.float32,
        )

        # We'll define a small mapping from atype -> pysc2 function
        # For demonstration, let's handle a few actions. 
        # If you need to handle all Terran actions, expand this dict.
        self.action_map = {
            0: actions.FUNCTIONS.no_op.id,
            1: actions.FUNCTIONS.select_army.id,
            2: actions.FUNCTIONS.Move_screen.id,
            32: actions.FUNCTIONS.select_point_screen.id,
            # Expand more if needed...
        }

        # Action space: MultiDiscrete([num_actions, 84, 84])
        # We have 33 possible atype (0..32), but we only define a subset here. 
        # If the user tries an undefined one, we do no_op fallback.
        self.num_action_types = 33
        self.action_space = gym.spaces.MultiDiscrete([
            self.num_action_types,
            84,
            84
        ])

    def reset(self, seed=None, options=None):
        del seed, options
        time_step = self.env.reset()[0]  # first agent's TimeStep
        obs = self._process_obs(time_step)
        return obs, {}

    def step(self, action: np.ndarray):
        try:
            # 1. Do a no-op to read the available actions
            no_op = actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
            timestep = self.env.step([no_op])[0]
            available_actions = timestep.observation["available_actions"]

            # 2. Parse the agent's chosen action
            atype = int(action[0])
            x = int(action[1])
            y = int(action[2])

            # 3. Convert (atype, x, y) -> SC2 FunctionCall
            sc2_action = self._discrete_to_sc2_action(atype, x, y, available_actions)

            # 4. Step environment
            step_result = self.env.step([sc2_action])
            time_step = step_result[0]

            # 5. Process obs
            obs = self._process_obs(time_step)
            done = time_step.last()

            # No custom reward for now, just 0.0
            return obs, 0.0, done, False, {}

        except Exception as e:
            print(f"Error in step execution: {e}")
            # Return a dummy obs, mark the episode done
            fallback_obs = np.zeros((6, 84, 84), dtype=np.float32)
            return fallback_obs, 0.0, True, False, {}

    def close(self):
        self.env.close()

    def _process_obs(self, time_step):
        """
        Returns (6,84,84) with these layers:
        unit_type, player_relative, selected, unit_hit_points,
        visibility_map, height_map
        """
        selected_layers = [
            features.SCREEN_FEATURES.unit_type.index,
            features.SCREEN_FEATURES.player_relative.index,
            features.SCREEN_FEATURES.selected.index,
            features.SCREEN_FEATURES.unit_hit_points.index,
            features.SCREEN_FEATURES.visibility_map.index,
            features.SCREEN_FEATURES.height_map.index,
        ]
        screen = time_step.observation["feature_screen"][selected_layers]
        return screen.astype(np.float32)

    def _discrete_to_sc2_action(self, atype, x, y, available_actions):
        """
        Convert (atype, x, y) => a valid SC2 FunctionCall.
        If action is not in available_actions, fallback to no_op.
        """

        # Default to no_op
        sc2_action = actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])

        # If atype is recognized in our action_map, pick that function ID
        func_id = self.action_map.get(atype, actions.FUNCTIONS.no_op.id)

        # Check if this function is actually available
        if func_id in available_actions:
            # Some actions require arguments, e.g. (queued=[0], [x, y])
            if func_id == actions.FUNCTIONS.select_army.id:
                # select_army => function args: [0] means "select all"  
                sc2_action = actions.FunctionCall(func_id, [[0]])
            elif func_id == actions.FUNCTIONS.Move_screen.id:
                # Move_screen => function args: [queued], [x, y]
                sc2_action = actions.FunctionCall(func_id, [[0], [x, y]])
            elif func_id == actions.FUNCTIONS.select_point_screen.id:
                # select_point_screen => function args: [type], [x,y]
                # type=0 => Replace existing selection
                sc2_action = actions.FunctionCall(func_id, [[0], [x, y]])
            else:
                # If we had a build or train action, we'd add them here
                sc2_action = actions.FunctionCall(func_id, [])
        else:
            # If not available, fallback to no_op to avoid errors
            sc2_action = actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])

        return sc2_action

##########################################
# 3. Train (New) or Resume an Existing Model
##########################################
if __name__ == "__main__":
    # Create the environment with multiple envs
    env = make_vec_env(lambda: SC2GymEnv(), n_envs=4)

    choice = input("Do you want to train a new model or resume an existing model? [new/resume]: ").strip().lower()

    if choice == "new":
        print("Creating a new model...")
        model = PPO(
            policy="CnnPolicy",
            env=env,
            verbose=1,
            device="cuda",  # or "cpu"
            n_steps=2048,
            tensorboard_log="./tb_logs",
            policy_kwargs={"normalize_images": False},
        )
    else:
        print("Resuming from existing model 'sc2_ppo_model'...")
        model = PPO.load("sc2_ppo_model", env=env, device="cuda")
        model.n_steps = 2048
        model.tensorboard_log = "./tb_logs"

    total_timesteps = 500000000
    timestep_chunk = 2048
    remaining_ts = total_timesteps

    try:
        iteration_count = 0
        while remaining_ts > 0:
            iteration_count += 1
            actual_ts = min(timestep_chunk, remaining_ts)

            model.learn(
                total_timesteps=actual_ts,
                reset_num_timesteps=False,
                tb_log_name="sc2_ppo"
            )

            remaining_ts -= actual_ts
            print(f"\nIteration {iteration_count} complete. Remaining timesteps: {remaining_ts}. Press 'q' to stop and save...")

            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == b'q':
                    print("\nEarly stopping triggered by user!")
                    break

    except KeyboardInterrupt:
        print("\nTraining interrupted by Ctrl+C! Saving model...")

    finally:
        model.save("sc2_ppo_model")
        print("Model saved to 'sc2_ppo_model'.")
        env.close()
