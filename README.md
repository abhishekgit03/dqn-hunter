# DeepRL — Blob Arena

A Deep Q-Network (DQN) reinforcement learning project where an AI agent learns to hunt prey in a 2D arena. Train the agent and watch it play in real-time through a browser-based visualizer.

![Tech Stack](https://img.shields.io/badge/PyTorch-DQN-orange) ![FastAPI](https://img.shields.io/badge/FastAPI-WebSocket-green) ![Frontend](https://img.shields.io/badge/Frontend-Vanilla_JS-yellow)

## Overview

The agent navigates an 800×600 arena trying to catch 50 intelligent prey. Prey exhibit flocking behavior — they flee from the agent, separate from neighbors, and bounce off walls. The agent uses a neural network to map observations to actions and learns through trial and error.

**Architecture:**
- **Agent:** 2-layer MLP (23-dim state → 128 → 128 → 16 actions)
- **Actions:** 16 discrete moves (8 compass directions × 2 speeds, plus idle)
- **State:** Position, velocity, 5 nearest prey positions/distances, wall distances
- **Rewards:** +10 per catch, time penalty, proximity bonus, wall penalty

## Project Structure

```
deeprl/
├── backend/
│   ├── agent.py          # DQN agent with replay buffer and target network
│   ├── environment.py    # BlobArena game environment
│   ├── train.py          # Training loop with checkpointing and logging
│   ├── main.py           # FastAPI WebSocket server for visualization
│   └── requirements.txt
└── frontend/
    ├── index.html        # Web UI with dual canvas layers
    ├── arena.js          # Agent/prey rendering via WebSocket
    └── heatmap.js        # Q-value heatmap visualization
```

## Setup

```bash
cd backend
python -m venv env
source env/bin/activate        # Windows: env\Scripts\activate
pip install -r requirements.txt
```

## Training

```bash
# Train from scratch
python train.py

# Resume from a checkpoint
python train.py --resume ep350

# Plot the reward curve
python train.py --plot
```

Checkpoints are saved every 50 episodes to `backend/checkpoints/`. Training logs are written to `backend/logs/training_log.csv`.

## Visualization

Start the server and open the visualizer in your browser:

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Visit [http://localhost:8000/static](http://localhost:8000/static).

The visualizer streams the game at 30 FPS over WebSocket. Every 10 frames, a 20×20 Q-value heatmap is computed and overlaid on the arena — purple regions are low-value, amber regions are high-value.

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| State dim | 23 |
| Action dim | 16 |
| Hidden layers | 128, 128 |
| Learning rate | 1e-4 |
| Discount (γ) | 0.95 |
| Replay buffer | 50,000 |
| Batch size | 64 |
| Epsilon decay | 0.997 (min 0.10) |
| Target net update (τ) | 0.005 |

## Dependencies

- [PyTorch](https://pytorch.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [uvicorn](https://www.uvicorn.org/)
- numpy
