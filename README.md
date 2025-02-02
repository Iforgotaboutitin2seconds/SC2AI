# StarCraft 2 Self-Learning AI

This project develops a self-learning AI for StarCraft 2 using a Deep Q-Network (DQN). The AI learns to play by interacting with the game environment and improving its strategy over time.

## Project Goals

* **Self-Learning:** The AI learns through reinforcement learning, specifically Q-learning, by interacting with the StarCraft 2 environment.
* **StarCraft 2 Compatibility:** The AI is designed to work with the StarCraft 2 game using the `pysc2` library.
* **Terran Race:** The AI agent plays as the Terran race.
* **Simple64 Map:** The AI is trained on the "Simple64" map.

## Development Environment

* **IDE:** Visual Studio (for code development and debugging).
* **Language:** Python.
* **StarCraft 2 API:** `pysc2` (DeepMind's Python interface to StarCraft II).
* **Dependencies:**
    * `pysc2`
    * `absl-py`
    * `torch` (PyTorch)
    * `numpy`

## Resources and Inspiration

This project draws inspiration and guidance from various AI learning resources, including (but not limited to):

* DeepMind's StarCraft II Learning Environment ([https://github.com/deepmind/pysc2](https://github.com/deepmind/pysc2))

## Current Implementation

The current implementation prioritizes simplicity and focuses on a basic DQN agent.  All code is contained within a single Python file:

* `train_sc2_ai.py`: This file contains the core logic for training the DQN agent, including the neural network architecture, experience replay, and the training loop.
