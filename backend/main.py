import os
import json
import asyncio
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from environment import BlobArena
from agent import DQNAgent

app = FastAPI()
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

CHECKPOINT_DIR = "checkpoints"
STATE_DIM      = 23
ACTION_DIM     = 16
FPS            = 30


def load_latest_checkpoint(agent):
    path = "checkpoints/dqn_ep350.pt"
    agent.load(path)
    return "dqn_ep350.pt"


def get_qvalue_heatmap(agent, env, grid=20):
    xs     = np.linspace(0, 1, grid)
    ys     = np.linspace(0, 1, grid)
    heatmap = []

    base_state = env._get_state().copy()

    for y in ys:
        row = []
        for x in xs:
            state    = base_state.copy()
            state[0] = x
            state[1] = y
            qvals = agent.get_q_values(state)
            row.append(round(float(np.max(qvals)), 3))
        heatmap.append(row)

    return heatmap


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    env   = BlobArena()
    agent = DQNAgent()
    agent.epsilon = 0.0

    checkpoint = load_latest_checkpoint(agent)
    await websocket.send_text(json.dumps({
        "type":       "info",
        "checkpoint": checkpoint or "none"
    }))

    state    = env.reset()
    heatmap_counter = 0

    try:
        while True:
            action = agent.act(state, training=False)
            state, reward, done, info = env.step(action)

            payload = {
                "type":    "state",
                "game":    env.to_json(),
                "reward":  round(reward, 3),
                "catches": int(info.get("catch", 0) / 10.0),
            }

            heatmap_counter += 1
            if heatmap_counter % 10 == 0:
                payload["heatmap"] = get_qvalue_heatmap(agent, env)

            await websocket.send_text(json.dumps(payload))

            if done:
                state = env.reset()

            await asyncio.sleep(1 / FPS)

    except WebSocketDisconnect:
        pass


@app.get("/reload")
def reload_checkpoint():
    return {"status": "reload triggered — reconnect WebSocket to apply"}