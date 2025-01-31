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
# 2. Define a Custom SC2 Env with Many Terran Units
###################################################################
class SC2GymEnv(gym.Env):
    """
    StarCraft II environment that:
      - Returns channel-first obs (3,84,84).
      - MultiDiscrete action space with ~24 action_types plus x,y coords.
      - Rewards:
         +0.5 / -0.3 for SCVs gained/lost
         +1 / -0.5 for Terran buildings gained/lost
         + (killed_minerals + killed_vespene)*0.001 for enemy kills
         +0.2 / -0.1 for Terran combat units gained/lost
    """

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

        ###############################
        # Observation: (3,84,84)
        ###############################
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(3, 84, 84),
            dtype=np.float32,
        )

        ################################################
        # Action Space: MultiDiscrete([num_actions, 84, 84])
        # We'll define ~24 action types:
        #   0 = no_op
        #   1 = select_army
        #   2 = move_screen
        #   3 = train SCV
        #   4 = build Supply Depot
        #   5 = build Barracks
        #   6 = build Refinery
        #   7 = build Factory
        #   8 = build Starport
        #   9  = train Marine
        #   10 = train Marauder
        #   11 = train Reaper
        #   12 = train Ghost
        #   13 = train Hellion
        #   14 = train WidowMine
        #   15 = train SiegeTank
        #   16 = train Thor
        #   17 = train Cyclone
        #   18 = train Viking
        #   19 = train Medivac
        #   20 = train Raven
        #   21 = train Banshee
        #   22 = train Battlecruiser
        #   23 = train Liberator
        ################################################
        self.num_action_types = 24
        self.action_space = gym.spaces.MultiDiscrete([self.num_action_types, 84, 84])

        ###############
        # SCV Tracking
        ###############
        self._old_scv_count = 0

        ###################################################
        # Terran Building Tracking
        ###################################################
        self.terran_buildings = {
            units.Terran.SupplyDepot: 0,
            units.Terran.Barracks: 0,
            units.Terran.Refinery: 0,
            units.Terran.Factory: 0,
            units.Terran.Starport: 0,
        }

        ###################################################
        # Terran Combat Unit Tracking
        # We'll track: Marine, Marauder, Reaper, Ghost,
        # Hellion, WidowMine, SiegeTank, Thor, Cyclone,
        # Viking, Medivac, Raven, Banshee, Battlecruiser, Liberator
        ###################################################
        self.terran_units = {
            units.Terran.Marine: 0,
            units.Terran.Marauder: 0,
            units.Terran.Reaper: 0,
            units.Terran.Ghost: 0,
            units.Terran.Hellion: 0,
            units.Terran.WidowMine: 0,
            units.Terran.SiegeTank: 0,
            units.Terran.Thor: 0,
            units.Terran.Cyclone: 0,
            units.Terran.VikingFighter: 0,  # Add VikingFighter
            units.Terran.VikingAssault: 0,  # Add VikingAssault
            units.Terran.Medivac: 0,
            units.Terran.Raven: 0,
            units.Terran.Banshee: 0,
            units.Terran.Battlecruiser: 0,
            units.Terran.Liberator: 0,
        }

        ###################################################
        # Score-based kill tracking
        ###################################################
        self._old_killed_minerals = 0
        self._old_killed_vespene = 0

    def reset(self, seed=None, options=None):
        del seed, options
        time_step = self.env.reset()[0]  # first agent's TimeStep

        # SCVs
        self._old_scv_count = self._count_scvs(time_step)

        # Buildings
        self._reset_building_counts(time_step)

        # Terran combat units
        self._reset_unit_counts(time_step)

        # Kills
        self._old_killed_minerals, self._old_killed_vespene = self._get_killed_resources(time_step)

        obs = self._process_obs(time_step)
        return obs, {}

    def step(self, action: np.ndarray):
        """
        action = [action_type, x, y], each in discrete range.
        We parse them, build an SC2 action, step the env, and compute custom rewards.
        """
        try:
            # 1. No-op to get available_actions
            no_op = actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
            timestep = self.env.step([no_op])[0]
            available_actions = timestep.observation["available_actions"]

            # 2. Convert (action_type, x, y) -> SC2 FunctionCall
            atype = int(action[0])
            x = int(action[1])
            y = int(action[2])

            sc2_action = self._discrete_to_sc2_action(atype, x, y, available_actions)

            # 3. Step environment
            step_result = self.env.step([sc2_action])
            time_step = step_result[0]

            # 4. Process obs
            obs = self._process_obs(time_step)

            # ========== CUSTOM REWARD ==========

            ### (A) SCVs
            new_scv_count = self._count_scvs(time_step)
            scv_diff = new_scv_count - self._old_scv_count
            scv_reward = 0.0
            if scv_diff > 0:
                scv_reward += scv_diff * 0.5
            elif scv_diff < 0:
                scv_reward += scv_diff * 0.3  # negative
            self._old_scv_count = new_scv_count

            ### (B) Buildings
            building_reward = self._check_building_changes(time_step)

            ### (C) Combat Units
            unit_reward = self._check_unit_changes(time_step)

            ### (D) Enemy Kills
            killed_m_diff, killed_v_diff = self._check_kills(time_step)
            kills_reward = 0.001 * (killed_m_diff + killed_v_diff)

            ### Combine
            custom_reward = scv_reward + building_reward + unit_reward + kills_reward
            # ===================================

            done = time_step.last()
            truncated = False
            info = {}

            return obs, custom_reward, done, truncated, info

        except Exception as e:
            print(f"Error in step execution: {e}")
            fallback_obs = np.zeros((3, 84, 84), dtype=np.float32)
            return fallback_obs, 0.0, True, False, {}

    def close(self):
        self.env.close()

    ##########################
    # Internal Helper Methods
    ##########################

    def _process_obs(self, time_step):
        """
        Extract (3,84,84) from the SC2 feature_screen in channel-first format.
        """
        selected_layers = [
            features.SCREEN_FEATURES.unit_hit_points.index,
            features.SCREEN_FEATURES.visibility_map.index,
            features.SCREEN_FEATURES.height_map.index,
        ]
        screen = time_step.observation["feature_screen"][selected_layers]  # shape: (3, 84, 84)
        screen = screen.astype(np.float32)
        return screen

    def _discrete_to_sc2_action(self, atype, x, y, available_actions):
        """
        Convert (atype, x, y) => a valid SC2 FunctionCall.

        0..8 = original actions (no-op,select_army,move,trainSCV,build..., etc.)
        9..23 = train Marine, Marauder, Reaper, Ghost, Hellion, WidowMine, SiegeTank,
                Thor, Cyclone, Viking, Medivac, Raven, Banshee, Battlecruiser, Liberator.
        """
        sc2_action = actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])

        # Helper function to create a "Train_X_quick" action if available
        def train_if_possible(train_id):
            if train_id in available_actions:
                return actions.FunctionCall(train_id, [[0]])
            return sc2_action  # default no-op if not available

        # Original 9
        if atype == 1:  # select_army
            if actions.FUNCTIONS.select_army.id in available_actions:
                sc2_action = actions.FunctionCall(actions.FUNCTIONS.select_army.id, [[0]])
        elif atype == 2:  # move_screen
            if actions.FUNCTIONS.Move_screen.id in available_actions:
                sc2_action = actions.FunctionCall(actions.FUNCTIONS.Move_screen.id, [
                    [0],  # not queued
                    [x, y]
                ])
        elif atype == 3:  # train SCV
            if actions.FUNCTIONS.Train_SCV_quick.id in available_actions:
                sc2_action = actions.FunctionCall(actions.FUNCTIONS.Train_SCV_quick.id, [[0]])
        elif atype == 4:  # build Supply Depot
            if actions.FUNCTIONS.Build_SupplyDepot_screen.id in available_actions:
                sc2_action = actions.FunctionCall(actions.FUNCTIONS.Build_SupplyDepot_screen.id, [
                    [0],
                    [x, y]
                ])
        elif atype == 5:  # build Barracks
            if actions.FUNCTIONS.Build_Barracks_screen.id in available_actions:
                sc2_action = actions.FunctionCall(actions.FUNCTIONS.Build_Barracks_screen.id, [
                    [0],
                    [x, y]
                ])
        elif atype == 6:  # build Refinery
            if actions.FUNCTIONS.Build_Refinery_screen.id in available_actions:
                sc2_action = actions.FunctionCall(actions.FUNCTIONS.Build_Refinery_screen.id, [
                    [0],
                    [x, y]
                ])
        elif atype == 7:  # build Factory
            if actions.FUNCTIONS.Build_Factory_screen.id in available_actions:
                sc2_action = actions.FunctionCall(actions.FUNCTIONS.Build_Factory_screen.id, [
                    [0],
                    [x, y]
                ])
        elif atype == 8:  # build Starport
            if actions.FUNCTIONS.Build_Starport_screen.id in available_actions:
                sc2_action = actions.FunctionCall(actions.FUNCTIONS.Build_Starport_screen.id, [
                    [0],
                    [x, y]
                ])

        # Additional 15 "train unit" actions:
        elif atype == 9:   # Marine
            sc2_action = train_if_possible(actions.FUNCTIONS.Train_Marine_quick.id)
        elif atype == 10:  # Marauder
            sc2_action = train_if_possible(actions.FUNCTIONS.Train_Marauder_quick.id)
        elif atype == 11:  # Reaper
            sc2_action = train_if_possible(actions.FUNCTIONS.Train_Reaper_quick.id)
        elif atype == 12:  # Ghost
            sc2_action = train_if_possible(actions.FUNCTIONS.Train_Ghost_quick.id)
        elif atype == 13:  # Hellion
            sc2_action = train_if_possible(actions.FUNCTIONS.Train_Hellion_quick.id)
        elif atype == 14:  # Widow Mine
            sc2_action = train_if_possible(actions.FUNCTIONS.Train_WidowMine_quick.id)
        elif atype == 15:  # Siege Tank
            sc2_action = train_if_possible(actions.FUNCTIONS.Train_SiegeTank_quick.id)
        elif atype == 16:  # Thor
            sc2_action = train_if_possible(actions.FUNCTIONS.Train_Thor_quick.id)
        elif atype == 17:  # Cyclone
            sc2_action = train_if_possible(actions.FUNCTIONS.Train_Cyclone_quick.id)
        elif atype == 18:  # Viking
            sc2_action = train_if_possible(actions.FUNCTIONS.Train_VikingFighter_quick.id)
        elif atype == 19:  # Medivac
            sc2_action = train_if_possible(actions.FUNCTIONS.Train_Medivac_quick.id)
        elif atype == 20:  # Raven
            sc2_action = train_if_possible(actions.FUNCTIONS.Train_Raven_quick.id)
        elif atype == 21:  # Banshee
            sc2_action = train_if_possible(actions.FUNCTIONS.Train_Banshee_quick.id)
        elif atype == 22:  # Battlecruiser
            sc2_action = train_if_possible(actions.FUNCTIONS.Train_Battlecruiser_quick.id)
        elif atype == 23:  # Liberator
            sc2_action = train_if_possible(actions.FUNCTIONS.Train_Liberator_quick.id)

        return sc2_action

    ############################
    # SCV Counting
    ############################
    def _count_scvs(self, time_step) -> int:
        """SCV = unit_type=45."""
        scv_unit_type = 45
        feature_units = time_step.observation["feature_units"]
        count = 0
        for unit in feature_units:
            if unit.unit_type == scv_unit_type:
                count += 1
        return count

    ###########################################
    # Terran Buildings: track changes
    ###########################################
    def _reset_building_counts(self, time_step):
        for b_type in self.terran_buildings.keys():
            self.terran_buildings[b_type] = 0

        feature_units = time_step.observation["feature_units"]
        for unit in feature_units:
            if unit.unit_type in self.terran_buildings:
                self.terran_buildings[unit.unit_type] += 1

    def _check_building_changes(self, time_step) -> float:
        """
        +1 for new building, -0.5 if building lost.
        """
        reward = 0.0
        new_counts = {b: 0 for b in self.terran_buildings}

        feature_units = time_step.observation["feature_units"]
        for unit in feature_units:
            if unit.unit_type in new_counts:
                new_counts[unit.unit_type] += 1

        for b_type in self.terran_buildings:
            diff = new_counts[b_type] - self.terran_buildings[b_type]
            if diff > 0:
                reward += diff * 1.0
            elif diff < 0:
                reward += diff * 0.5  # negative
            self.terran_buildings[b_type] = new_counts[b_type]

        return reward

    ###########################################
    # Terran Combat Units
    ###########################################
    def _reset_unit_counts(self, time_step):
        for u_type in self.terran_units.keys():
            self.terran_units[u_type] = 0

        feature_units = time_step.observation["feature_units"]
        for unit in feature_units:
            if unit.unit_type in self.terran_units:
                self.terran_units[unit.unit_type] += 1

    def _check_unit_changes(self, time_step) -> float:
        """
        +0.2 for new Terran unit, -0.1 if unit lost.
        """
        reward = 0.0
        new_counts = {u: 0 for u in self.terran_units}

        feature_units = time_step.observation["feature_units"]
        for unit in feature_units:
            if unit.unit_type in new_counts:
                new_counts[unit.unit_type] += 1

        for u_type in self.terran_units:
            diff = new_counts[u_type] - self.terran_units[u_type]
            if diff > 0:
                reward += diff * 0.2
            elif diff < 0:
                reward += diff * 0.1  # negative
            self.terran_units[u_type] = new_counts[u_type]

        return reward

    ###########################################
    # Enemy Kills: from "score_by_category"
    ###########################################
    def _get_killed_resources(self, time_step) -> tuple:
        """
        Returns (killed_minerals, killed_vespene) from score_by_category.
        row=1 => killed_minerals, row=2 => killed_vespene
        sum across the 5 categories.
        """
        score_by_category = time_step.observation["score_by_category"]  # shape=(11,5)
        killed_m = np.sum(score_by_category[1, :])
        killed_v = np.sum(score_by_category[2, :])
        return killed_m, killed_v

    def _check_kills(self, time_step) -> tuple:
        """Compare old vs new killed_minerals+vespene, return (dm, dv)."""
        killed_m_new, killed_v_new = self._get_killed_resources(time_step)
        dm = killed_m_new - self._old_killed_minerals
        dv = killed_v_new - self._old_killed_vespene
        self._old_killed_minerals = killed_m_new
        self._old_killed_vespene = killed_v_new
        return dm, dv


##########################################
# 3. Train (New) or Resume an Existing Model
##########################################
if __name__ == "__main__":
    env = make_vec_env(lambda: SC2GymEnv(), n_envs=1)

    # (A) Start from scratch
    model = PPO(
        policy="CnnPolicy",
        env=env,
        verbose=1,
        device="cuda",  # or "cpu"
        policy_kwargs={"normalize_images": False},
    )

    # (B) If you want to resume from a matching model:
    # model = PPO.load("sc2_ppo_model", env=env, device="cuda")

    total_timesteps = 500000000
    timestep_chunk = 1000  # Check for keypress every 1000 timesteps
    remaining_ts = total_timesteps

    try:
        while remaining_ts > 0:
            # Train in chunks and check for keypress between chunks
            actual_ts = min(timestep_chunk, remaining_ts)
            model.learn(actual_ts, reset_num_timesteps=False)
            remaining_ts -= actual_ts
            
            print(f"Remaining timesteps: {remaining_ts}. Press 'q' to stop and save...")
            
            # Check for 'q' keypress
            if msvcrt.kbhit():  # Check if key pressed
                key = msvcrt.getch()  # Get the key
                if key == b'q':  # b'q' for bytes comparison
                    print("\nEarly stopping triggered by user!")
                    break

    except KeyboardInterrupt:
        print("\nTraining interrupted by Ctrl+C! Saving model...")

    finally:
        # Always save when exiting
        model.save("sc2_ppo_model")
        print("Model saved to 'sc2_ppo_model'.")
        env.close()