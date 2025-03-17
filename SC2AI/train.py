import sc2
from sc2.bot_ai import BotAI
from sc2.data import Race, Difficulty
from sc2.main import run_game
from sc2.player import Bot, Computer
import numpy as np
import random
import time

# Choose your machine learning framework
# import tensorflow as tf
# from tensorflow.keras import layers
import torch
import torch.nn as nn
import torch.optim as optim

class TerranAgent(BotAI):
    def __init__(self, model=None, optimizer=None, epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.01, gamma=0.99):
        self.model = model
        self.optimizer = optimizer
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.memory = [] # For storing experiences (state, action, reward, next_state, done)
        self.n_actions = self.get_available_actions_count() # You'll need to implement this

    def on_start(self):
        print("Game started!")

    async def on_step(self, iteration: int):
        await self.distribute_workers()  # Basic worker distribution

        if self.model is None:
            # Initialize a simple model if not provided
            self.initialize_model()

        if iteration % 5 == 0: # Example: Decide action every few steps
            state = self.get_state()
            action = self.choose_action(state)
            await self.execute_action(action)

            # For demonstration, let's just build a supply depot if we have less than 13 supply
            if self.supply_left < 3 and self.can_afford(self.game_data.units[Race.Terran][0]): # SupplyDepot
                await self.build(self.game_data.units[Race.Terran][0], self.start_location.towards(self.game_info.map_center, 5))

    def initialize_model(self):
        # Example using PyTorch
        class SimpleModel(nn.Module):
            def __init__(self, input_size, num_actions):
                super(SimpleModel, self).__init__()
                self.fc1 = nn.Linear(input_size, 64)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(64, num_actions)

            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                return x

        # Determine the input size based on your state representation
        input_size = self.get_state_size()
        self.n_actions = self.get_available_actions_count()
        self.model = SimpleModel(input_size, self.n_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def get_state(self):
        # Implement your state representation here
        # This could include things like:
        # - Number of units of each type
        # - Available resources (minerals, gas)
        # - Supply counts
        # - Enemy unit counts (if visible)
        return np.array([
            len(self.units(Race.Terran).ready),
            self.minerals,
            self.vespene,
            self.supply_used,
            self.supply_cap
            # Add more features as needed
        ], dtype=np.float32)

    def get_state_size(self):
        return len(self.get_state())

    def get_available_actions_count(self):
        # This is a crucial part. You need to define the set of actions your AI can take.
        # For a beginner AI, you might start with a very limited set of actions like:
        # - Build Supply Depot
        # - Build Barracks
        # - Train Marine
        # - Attack with all units
        # - Do nothing
        # The number of possible actions will be the size of your action space.
        # For now, let's just return a placeholder.
        return 5 # Example number of actions

    def choose_action(self, state):
        if random.random() < self.epsilon:
            # Explore: choose a random action
            return random.randrange(self.n_actions)
        else:
            # Exploit: choose the best action based on the model's prediction
            if self.model is not None:
                self.model.eval()
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    q_values = self.model(state_tensor)
                    return torch.argmax(q_values).item()
            else:
                # If the model is not initialized, choose a random action
                return random.randrange(self.n_actions)

    async def execute_action(self, action):
        # Implement the logic for executing the chosen action
        # This will involve using the sc2 API to issue commands.
        if action == 0:
            # Example: Try to build a Supply Depot
            if self.can_afford(self.game_data.units[Race.Terran][0]): # SupplyDepot
                await self.build(self.game_data.units[Race.Terran][0], self.start_location.towards(self.game_info.map_center, 5))
        elif action == 1:
            # Example: Try to build a Barracks
            if self.can_afford(self.game_data.units[Race.Terran][1]): # Barracks
                await self.build(self.game_data.units[Race.Terran][1], self.start_location.towards(self.game_info.map_center, 8))
        # Add more actions here

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(list(actions), dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(list(rewards), dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(list(dones), dtype=torch.float32).unsqueeze(1)

        self.model.train()
        q_values = self.model(states)
        q_value = q_values.gather(1, actions)

        next_q_values = self.model(next_states).max(1)[0].unsqueeze(1)
        target_q_value = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_value, target_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

async def main():
    agent = TerranAgent()
    try:
        await run_game(
            [Bot(Race.Terran, agent),
             Computer(Race.Random, Difficulty.Easy)],
            sc2.maps.get("CatalystLE"), # Choose a map
            realtime=False # Set to True for faster debugging, False for training
        )
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Game ended!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())