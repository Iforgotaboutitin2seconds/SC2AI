# StarCraft 2 AI - Do Nothing Terran Bot

A simple StarCraft 2 bot using the [python-sc2](https://github.com/BurnySc2/python-sc2) library. This bot:
- Plays as Terran
- Currently does nothing (placeholder for future development)
- Plays against a built-in Easy Zerg AI
- Uses the Simple64 map

## Requirements

- Python 3.6+
- StarCraft 2 installed
- python-sc2 library

## Setup

1. Install the required dependencies:
```
pip install -r requirements.txt
```

2. Make sure StarCraft 2 is installed on your system.

## Running the AI

Run the bot with:
```
python terran_ai.py
```

This will launch StarCraft 2 and start a game with your do-nothing Terran bot against the built-in Easy Zerg AI on the Simple64 map.

## Development

You can extend the functionality of the bot by modifying the `on_step` method in the `DoNothingBot` class in `terran_ai.py`. 