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
```
---

## 🛠️Installation Instructions
### 1. Clone the repo
```bash
git clone https://github.com/rchimome/Crafter.git
cd Crafter_Project
```
### 2. Set up virtual environment (optional but recommended)
```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
If using gymnasium, make sure to also install:
```bash
pip install gym==0.21.0 stable-baselines3[extra]
```

---

## 📌 Notes
- All training uses 500,000 timesteps per agent iteration.
- Models are saved in .zip or .pth depending on implementation.
- Logs are grouped by iteration (logs/a2c_imp1/, etc.) for easy evaluation comparison.

---

## 👨‍🎓 Contributors
- Christopher Musiiwa — Student Number: 707982
---
## 📎 License
- This project is for academic use only as part of the MSc AI Reinforcement Learning coursework at Wits University (2026).
---
