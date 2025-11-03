# utils.py
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.evaluation import evaluate_policy
from crafter import Recorder
from shimmy import GymV21CompatibilityV0
from gym.envs.registration import register
import crafter
import gym as old_gym
# utils.py

def make_env(logdir):
    try:
        register(id='CrafterNoReward-v1', entry_point=crafter.Env)
    except Exception:
        pass  # already registered

    # Create raw gym-compatible environment
    env = old_gym.make('CrafterNoReward-v1')

    # NOTE: Do NOT wrap here — return raw env to DummyVecEnv
    return env


def plot_eval_metrics(logdirs, labels, outdir="plots"):
    os.makedirs(outdir, exist_ok=True)
    all_rewards, all_lengths, all_achievements = [], [], []

    for logdir in logdirs:
        rewards, lengths, achievements = [], [], []
        for run in os.listdir(logdir):
            path = os.path.join(logdir, run, "stats.npy")
            if os.path.isfile(path):
                stats = np.load(path, allow_pickle=True).item()
                rewards.append(stats['return'])
                lengths.append(stats['length'])
                achievements.append(len(stats['achievements']))
        all_rewards.append(np.mean(rewards))
        all_lengths.append(np.mean(lengths))
        all_achievements.append(np.mean(achievements))

    x = np.arange(len(labels))

    # Reward
    plt.figure()
    plt.bar(x, all_rewards)
    plt.xticks(x, labels)
    plt.ylabel("Reward")
    plt.title("Average Reward")
    plt.savefig(f"{outdir}/reward_plot.png")

    # Survival
    plt.figure()
    plt.bar(x, all_lengths)
    plt.xticks(x, labels)
    plt.ylabel("Steps")
    plt.title("Average Survival Time")
    plt.savefig(f"{outdir}/survival_plot.png")

    # Achievements
    plt.figure()
    plt.bar(x, all_achievements)
    plt.xticks(x, labels)
    plt.ylabel("Achievements")
    plt.title("Average Achievements")
    plt.savefig(f"{outdir}/achievements_plot.png")
