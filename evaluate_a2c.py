import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from utils import make_env

# Paths to model checkpoints
model_paths = {
    "A2C Baseline": r"C:\Users\ChristopherMusiiwa\Reinforcement-Learning-Project-2026-Crafter\a2c_baseline.zip",
    "A2C + Reward Shaping": r"C:\Users\ChristopherMusiiwa\Reinforcement-Learning-Project-2026-Crafter\a2c_imp1.zip",
    "A2C + Frame Stack": r"C:\Users\ChristopherMusiiwa\Reinforcement-Learning-Project-2026-Crafter\a2c_imp2.zip",
}

# Evaluation settings
n_eval_episodes = 10
results = {}

# Evaluate each model
for label, model_path in model_paths.items():
    print(f"\nEvaluating: {label}")
    logdir = os.path.splitext(os.path.basename(model_path))[0]
    
    # Environment setup
    env = DummyVecEnv([lambda: make_env(f"logs/{logdir}")])
    
    # Load model
    model = A2C.load(model_path, env=env)
    
    # Evaluation loop
    episode_rewards = []
    for episode in range(n_eval_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}")

    results[label] = episode_rewards

# Save raw metrics
for label, rewards in results.items():
    np.savez(f"logs/{label.replace(' ', '_').lower()}_eval.npz", rewards=rewards)

# Plotting
plt.figure(figsize=(12, 6))
for label, rewards in results.items():
    plt.plot(rewards, marker='o', label=f"{label} (Avg: {np.mean(rewards):.2f})")

plt.title("A2C Model Comparison (Total Reward per Episode)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("logs/a2c_comparison_plot.png")
plt.show()
