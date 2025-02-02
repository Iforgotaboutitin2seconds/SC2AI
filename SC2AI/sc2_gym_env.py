import gym
import numpy as np
from gym import spaces
from pysc2.env import sc2_env
from pysc2.lib import actions, features
from absl import flags

FLAGS = flags.FLAGS
if not FLAGS.is_parsed():
    FLAGS(["sc2_gym_env.py"])

class SC2GymEnv(gym.Env):
    def __init__(self):
        super(SC2GymEnv, self).__init__()

        self.env = sc2_env.SC2Env(
            map_name="Simple64",
            players=[
                sc2_env.Agent(sc2_env.Race.terran),
                sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.hard)
            ],
            agent_interface_format=sc2_env.AgentInterfaceFormat(
                feature_dimensions=sc2_env.Dimensions(screen=64, minimap=64),
                use_feature_units=True
            ),
            step_mul=8,
            game_steps_per_episode=0,
            visualize=False
        )

        # ACTION SPACE (Function ID, Target X, Target Y, Select Index)
        self.action_space = spaces.MultiDiscrete([len(actions.FUNCTIONS), 64, 64, 10])

        # OBSERVATION SPACE (Minimap + Unit Features)
        self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)

    def reset(self):
        timestep = self.env.reset()[0]
        self.current_obs = timestep
        return self._process_obs(timestep)

    def step(self, action):
        action_id, x, y, select_index = action  # Unpack action tuple

        sc2_action = self._map_action(action_id, (x, y), select_index)
        timestep = self.env.step([sc2_action])[0]
        
        reward = timestep.reward
        done = timestep.last()

        return self._process_obs(timestep), reward, done, {}

    def _process_obs(self, timestep):
        minimap = timestep.observation["feature_minimap"]
        screen = timestep.observation["feature_screen"]
        
        # Extract useful feature layers
        obs = np.stack([
            minimap[features.MINIMAP_FEATURES.player_relative.index], 
            screen[features.SCREEN_FEATURES.unit_type.index],
            screen[features.SCREEN_FEATURES.selected.index]  # Helps track selected unit
        ], axis=-1)
        
        self.current_obs = timestep
        return obs

    def _map_action(self, action_id, target, select_index):
        available_actions = self.current_obs.observation["available_actions"]
        units = self.current_obs.observation["feature_units"]
        x, y = target

        # Check if the action is available
        if action_id not in available_actions:
            return actions.FUNCTIONS.no_op()

        # Select a unit if needed
        if action_id == actions.FUNCTIONS.select_point.id and len(units) > select_index:
            unit = units[select_index]
            return actions.FUNCTIONS.select_point("select", (unit.x, unit.y))

        # Handling move, attack, and build actions
        if action_id == actions.FUNCTIONS.Move_screen.id:
            return actions.FUNCTIONS.Move_screen("queued", (x, y))

        elif action_id == actions.FUNCTIONS.Attack_screen.id:
            return actions.FUNCTIONS.Attack_screen("queued", (x, y))

        elif action_id == actions.FUNCTIONS.Build_Barracks_screen.id:
            return actions.FUNCTIONS.Build_Barracks_screen("queued", (x, y))

        return actions.FUNCTIONS.no_op()

    def close(self):
        self.env.close()
