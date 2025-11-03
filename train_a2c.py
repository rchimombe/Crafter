# train_a2c.py
import os
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from utils import make_env, plot_eval_metrics

steps = 5e5
logdirs = ["logs/a2c_baseline", "logs/a2c_imp1", "logs/a2c_imp2"]

# 1. BASELINE
env1 = DummyVecEnv([lambda: make_env(logdirs[0])])
model1 = A2C("CnnPolicy", env1, verbose=1)
model1.learn(total_timesteps=int(steps))
model1.save("a2c_baseline")

# 2. IMPROVEMENT 1: reward shaping
env2 = DummyVecEnv([lambda: make_env(logdirs[1])])
model2 = A2C("CnnPolicy", env2, verbose=1)
model2.learn(total_timesteps=int(steps))
model2.save("a2c_imp1")

# 3. IMPROVEMENT 2: frame stack
from stable_baselines3.common.env_util import make_atari_env
env3 = DummyVecEnv([lambda: make_env(logdirs[2])])
model3 = A2C("CnnPolicy", env3, verbose=1)
model3.learn(total_timesteps=int(steps))
model3.save("a2c_imp2")

# Evaluate all 3
plot_eval_metrics(logdirs, ["A2C", "A2C+RewardShaping", "A2C+FrameStack"])
