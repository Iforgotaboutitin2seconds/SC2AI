import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import *
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import time
import os
import signal
import sys

class TerranTrainingBot(sc2.BotAI):
    # Class variable for model path
    MODEL_PATH = 'terran_ai_model.h5'

    def __init__(self):
        super().__init__()
        self.model = self.load_or_create_model()
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32

    def load_or_create_model(self):
        if os.path.exists(self.MODEL_PATH):
            print("Loading existing model...")
            try:
                model = tf.keras.models.load_model(self.MODEL_PATH)
                print("Model loaded successfully")
                self.epsilon = self.epsilon_min  # Start with minimal exploration if loading trained model
                return model
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Creating new model instead")
        
        # Create new model if loading fails or no model exists
        print("Creating new model...")
        model = tf.keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(8,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(5, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    async def on_step(self, iteration):
        await self.distribute_workers()
        state = self.get_game_state()
        action = self.choose_action(state)
        reward = await self.execute_action(action)
        next_state = self.get_game_state()
        
        self.memory.append((state, action, reward, next_state))
        
        if len(self.memory) >= self.batch_size:
            self.train_model()

    def get_game_state(self):
        return np.array([
            self.minerals,
            self.vespene,
            self.supply_used / max(self.supply_cap, 1),
            len(self.units(SCV)),
            len(self.units(COMMANDCENTER)),
            len(self.units(BARRACKS)),
            len(self.units(MARINE)),
            self.time / 60
        ])

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 4)
        else:
            q_values = self.model.predict(state.reshape(1, -1), verbose=0)
            return np.argmax(q_values[0])

    async def execute_action(self, action):
        reward = 0
        if action == 0:  # Build SCV
            if self.can_afford(SCV) and self.units(COMMANDCENTER).ready.exists:
                cc = self.units(COMMANDCENTER).ready.random
                await self.do(cc.train(SCV))
                reward = 10
        elif action == 1:  # Build Supply Depot
            if self.can_afford(SUPPLYDEPOT):
                await self.build(SUPPLYDEPOT, near=self.units(COMMANDCENTER).first.position.towards(self.game_info.map_center, 8))
                reward = 15
        elif action == 2:  # Build Barracks
            if self.can_afford(BARRACKS):
                await self.build(BARRACKS, near=self.units(COMMANDCENTER).first.position.towards(self.game_info.map_center, 10))
                reward = 20
        elif action == 3:  # Train Marine
            if self.can_afford(MARINE) and self.units(BARRACKS).ready.exists:
                barracks = self.units(BARRACKS).ready.random
                await self.do(barracks.train(MARINE))
                reward = 15
        elif action == 4:  # Attack
            if self.units(MARINE).amount > 5:
                for marine in self.units(MARINE).idle:
                    await self.do(marine.attack(self.enemy_start_locations[0]))
                reward = 25
        
        return reward

    def train_model(self):
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([experience[0] for experience in batch])
        next_states = np.array([experience[3] for experience in batch])
        
        q_values = self.model.predict(states, verbose=0)
        next_q_values = self.model.predict(next_states, verbose=0)
        
        for i, (state, action, reward, next_state) in enumerate(batch):
            target = reward + self.gamma * np.max(next_q_values[i])
            q_values[i][action] = target
        
        self.model.fit(states, q_values, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    async def on_end(self, game_result):
        self.save_model()

    def save_model(self):
        try:
            self.model.save(self.MODEL_PATH)
            print(f"Model saved to {self.MODEL_PATH}")
        except Exception as e:
            print(f"Error saving model: {e}")

# Global flag for stopping training
keep_training = True

def signal_handler(sig, frame):
    global keep_training
    print('\nReceived interrupt signal. Finishing current game and saving model...')
    keep_training = False

def train_ai(episodes=100):
    global keep_training
    episode = 0
    
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    while episode < episodes and keep_training:
        print(f"Starting Episode {episode + 1}/{episodes}")
        bot = TerranTrainingBot()
        try:
            run_game(
                maps.get("Simple64"),
                [
                    Bot(Race.Terran, bot),
                    Computer(Race.Random, Difficulty.Easy)
                ],
                realtime=False,
                save_replay_as=f"replays/replay_{episode}.SC2Replay"
            )
            episode += 1
        except Exception as e:
            print(f"Error during game: {e}")
        
        # Save model after each episode
        bot.save_model()
        time.sleep(1)
    
    if not keep_training:
        print("Training stopped by user")
    else:
        print("Training completed")

if __name__ == '__main__':
    # Dependencies: pip install sc2 tensorflow numpy
    
    # Create replay directory
    if not os.path.exists('replays'):
        os.makedirs('replays')
    
    try:
        train_ai(episodes=100)
    except Exception as e:
        print(f"Training interrupted with error: {e}")
    finally:
        print("Training session ended")