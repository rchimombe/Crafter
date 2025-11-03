# 🧠 Crafter Reinforcement Learning Project (A2C & Recurrent PPO)

This project implements and evaluates two reinforcement learning agents — **A2C** and **Recurrent PPO (CnnLSTM)** — in the **Crafter** environment. The aim is to iteratively improve each agent and compare their performance using standard Crafter metrics.

---

## 📁 Project Structure

```bash
Reinforcement-Learning-Project-2026-Crafter/
│
├── train_a2c.py               # Training script for A2C agent with 2 improvements
├── train_rppo.py              # Training script for Recurrent PPO with 2 improvements
├── eval_a2c.py                # Evaluation script for A2C models
├── eval_rppo.py               # Evaluation script for RPPO models
├── utils.py                   # Shared utilities (env wrapper, plotting, reward shaping, etc.)
│
├── models/                    # Saved models (.zip or .pth)
│   ├── a2c_baseline.zip
│   ├── a2c_imp1.zip
│   ├── a2c_imp2.zip
│   ├── rppo_baseline.zip
│   ├── rppo_imp1.zip
│   └── rppo_imp2.zip
│
├── logs/                      # Logging folders for evaluation and plots
│   ├── a2c_baseline/
│   ├── a2c_imp1/
│   ├── a2c_imp2/
│   ├── rppo_baseline/
│   ├── rppo_imp1/
│   └── rppo_imp2/
│
├── report.pdf                 # Final written report for the assignment
└── README.md                  # Project guide (this file)
# Crafter RL Project – PPO & A2C Training

This project demonstrates **training and evaluating two reinforcement learning algorithms** on the [Crafter](https://github.com/danijar/crafter) environment using [Stable-Baselines3](https://stable-baselines3.readthedocs.io/).

The goal is to teach students how to:
- Train an RL agent on a **Partially observable environment** 
- Test generalization on **unseen environments**.
- Compare performance across algorithms to understand **robustness and generalization**.

---

## 🧠 Learning Objectives

By the end of this project, students should be able to:
- Configure and run **two RL algorithms** on the same environment.
- Analyze results by comparing performance across **seen** and **unseen seeds**(optional).

---

## 🛠 Setup

### 1. Clone the Project
```bash
git clone https://github.com/rayrsys/Reinforcement-Learning-Project-2026-Crafter.git
cd Crafter_Project
start coding ;)

