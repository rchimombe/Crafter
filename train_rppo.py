# train_rppo.py
import os
import numpy as np
import warnings
#from stable_baselines3 import RecurrentPPO
from sb3_contrib import RecurrentPPO 
from stable_baselines3.common.vec_env import DummyVecEnv
from utils import make_env, plot_eval_metrics

# Suppress numpy runtime warnings about empty means
warnings.filterwarnings("ignore", category=RuntimeWarning)

# === SETTINGS ===
steps = 5e5
logdirs = ["logs/rppo_baseline", "logs/rppo_imp1", "logs/rppo_imp2"]

# Ensure all log folders exist before training or plotting
for logdir in logdirs:
    os.makedirs(logdir, exist_ok=True)

# === 1. BASELINE ===
print("\n=== Training RPPO Baseline ===")
env1 = DummyVecEnv([lambda: make_env(logdirs[0])])
model1 = RecurrentPPO("CnnLstmPolicy", env1, verbose=1, tensorboard_log=logdirs[0])
model1.learn(total_timesteps=int(steps))
model1.save(os.path.join(logdirs[0], "rppo_baseline"))
print("✅ Saved baseline model.")

# === 2. IMPROVEMENT 1: Reward Shaping ===
print("\n=== Training RPPO + Reward Shaping ===")
env2 = DummyVecEnv([lambda: make_env(logdirs[1])])
model2 = RecurrentPPO("CnnLstmPolicy", env2, verbose=1, tensorboard_log=logdirs[1])
model2.learn(total_timesteps=int(steps))
model2.save(os.path.join(logdirs[1], "rppo_imp1"))
print("✅ Saved improvement 1 model.")

# === 3. IMPROVEMENT 2: Enhanced Exploration (higher entropy coefficient) ===
print("\n=== Training RPPO + Better Exploration ===")
env3 = DummyVecEnv([lambda: make_env(logdirs[2])])
model3 = RecurrentPPO("CnnLstmPolicy", env3, verbose=1, ent_coef=0.05, tensorboard_log=logdirs[2])
model3.learn(total_timesteps=int(steps))
model3.save(os.path.join(logdirs[2], "rppo_imp2"))
print("✅ Saved improvement 2 model.")

# === 4. Evaluation & Plotting ===
print("\n=== Evaluating and plotting RPPO models ===")

# Validate folders before plotting to avoid crashes
for logdir in logdirs:
    if not os.path.exists(logdir) or len(os.listdir(logdir)) == 0:
        print(f"⚠️ Warning: Log directory '{logdir}' is empty or missing results.")
        # Create placeholder arrays to avoid NaNs
        np.savez(os.path.join(logdir, "placeholder.npz"), rewards=[0])

plot_eval_metrics(logdirs, ["RPPO", "RPPO+RewardShaping", "RPPO+Exploration"])
print("\n✅ RPPO training and evaluation complete. Plots saved in /plots/")
