import os
import gym
import crafter
import numpy as np
import matplotlib.pyplot as plt

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from shimmy import GymV21CompatibilityV0

# --- Helper: Make the environment ---
def make_env():
    env = gym.make("CrafterNoReward-v1")
    env = crafter.Recorder(
        env,
        "./eval_logs",
        save_stats=False,
        save_video=False,
        save_episode=False,
    )
    return GymV21CompatibilityV0(env=env)

# --- Helper: Evaluate an agent ---
def evaluate_model(path_to_zip, n_episodes=10):
    model = RecurrentPPO.load(path_to_zip)
    env = DummyVecEnv([make_env])
    
    episode_rewards = []

    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        state = None  # For LSTM memory
        total_reward = 0.0
        while not done:
            action, state = model.predict(obs, state=state, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
        episode_rewards.append(total_reward)

    return episode_rewards

# --- Paths to models ---
model_paths = {
    "RPPO Baseline": "logs/rppo_baseline/rppo_baseline.zip",
    "RPPO + Reward Shaping": "logs/rppo_imp1/rppo_imp1.zip",
    "RPPO + Exploration": "logs/rppo_imp2/rppo_imp2.zip"
}

# --- Run evaluation ---
results = {}
for label, path in model_paths.items():
    print(f"Evaluating {label}...")
    rewards = evaluate_model(path, n_episodes=20)
    results[label] = rewards

# --- Plotting ---
plt.figure(figsize=(10, 6))
for label, rewards in results.items():
    plt.plot(rewards, label=label, marker='o')

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Reward per Episode (Recurrent PPO Variants)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("rppo_eval_plot.png")
plt.show()
