import os
import csv
import time
import argparse
import numpy as np

from environment import BlobArena
from agent import DQNAgent

CONFIG = {
    "total_episodes": 2000,
    "max_steps":      2000,
    "log_every":      10,
    "save_every":     50,
    "log_dir":        "logs",
    "checkpoint_dir": "checkpoints",
}


def ensure_dirs():
    os.makedirs(CONFIG["log_dir"],        exist_ok=True)
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)


def make_csv_writer(path):
    f = open(path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["episode", "reward", "catches", "steps", "epsilon", "loss", "buffer_size", "elapsed_min"])
    f.flush()
    return f, writer


def train(resume_from=None):
    ensure_dirs()

    env   = BlobArena()
    agent = DQNAgent()

    start_episode = 1
    if resume_from:
        path = os.path.join(CONFIG["checkpoint_dir"], f"dqn_{resume_from}.pt")
        agent.load(path)
        try:
            start_episode = int(resume_from.replace("ep", "")) + 1
        except ValueError:
            pass
        print(f"Resumed from {path}, starting at episode {start_episode}")

    log_path = os.path.join(CONFIG["log_dir"], "training_log.csv")
    log_file, csv_writer = make_csv_writer(log_path)

    print(f"{'Ep':>6} {'Reward':>9} {'Catches':>8} {'Steps':>7} {'e':>7} {'Loss':>9}")
    print("-" * 55)

    t0             = time.time()
    recent_rewards = []
    recent_catches = []
    recent_losses  = []

    for episode in range(start_episode, start_episode + CONFIG["total_episodes"]):
        state      = env.reset()
        ep_reward  = 0.0
        ep_catches = 0
        ep_losses  = []

        for step in range(CONFIG["max_steps"]):
            action                         = agent.act(state)
            next_state, reward, done, info = env.step(action)

            agent.remember(state, action, reward, next_state, done)

            loss = agent._learn()
            if loss is not None:
                ep_losses.append(loss)

            ep_reward  += reward
            ep_catches += int(info.get("catch", 0) / 5.0)
            state       = next_state

            if done:
                break

        agent.decay_epsilon()

        recent_rewards.append(ep_reward)
        recent_catches.append(ep_catches)
        recent_losses.append(float(np.mean(ep_losses)) if ep_losses else 0.0)

        if len(recent_rewards) > 20:
            recent_rewards.pop(0)
            recent_catches.pop(0)
            recent_losses.pop(0)

        if episode % CONFIG["log_every"] == 0:
            avg_r  = np.mean(recent_rewards)
            avg_c  = np.mean(recent_catches)
            avg_l  = np.mean(recent_losses)
            marker = ""
            if avg_c > 0.5: marker = "  <- learning"
            if avg_c > 2.0: marker = "  <- getting good"
            if avg_c > 4.0: marker = "  <- scary good"
            print(f"{episode:>6} {avg_r:>9.1f} {avg_c:>8.2f} {step+1:>7} {agent.epsilon:>7.4f} {avg_l:>9.5f}{marker}")

        csv_writer.writerow([
            episode, round(ep_reward, 2), ep_catches, step + 1,
            round(agent.epsilon, 5), round(recent_losses[-1], 6),
            len(agent.buffer), round((time.time() - t0) / 60, 2)
        ])
        log_file.flush()

        if episode % CONFIG["save_every"] == 0:
            agent.save(episode)

    final_ckpt = os.path.join(CONFIG["checkpoint_dir"], "dqn_final.pt")
    agent.save(final_ckpt)
    log_file.close()
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Final checkpoint: {final_ckpt}")


def plot_log(csv_path="logs/training_log.csv"):
    if not os.path.exists(csv_path):
        print("No log file found.")
        return

    rewards = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rewards.append(float(row["reward"]))

    if not rewards:
        print("Log file is empty.")
        return

    smoothed = [np.mean(rewards[max(0, i-20):i+1]) for i in range(len(rewards))]
    mn, mx   = min(smoothed), max(smoothed)
    height   = 12
    width    = min(80, len(smoothed))
    sampled  = smoothed[::max(1, len(smoothed) // width)][:width]

    print(f"\nReward curve ({len(rewards)} episodes)  min:{mn:.0f}  max:{mx:.0f}\n")
    for row in range(height, -1, -1):
        threshold = mn + (mx - mn) * row / height
        line      = "".join("X" if v >= threshold else " " for v in sampled)
        label     = f"{threshold:>8.0f} |" if row % 3 == 0 else "         |"
        print(label + line)
    print("         +" + "-" * len(sampled))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--plot",   action="store_true")
    args = parser.parse_args()

    if args.plot:
        plot_log()
    else:
        train(resume_from=args.resume)