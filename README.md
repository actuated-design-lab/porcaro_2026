# Porcaro 2026: Embodied AI Dual-Arm Drumming Project

This repository contains the official implementation of the paper:
*"Embodied Drumming: Sim-to-Real Reinforcement Learning for Pneumatic Musculoskeletal Robots via Generalized Rhythm Modeling"* 

This project simulates and trains **Porcaro**, a drumming robot driven by Pneumatic Artificial Muscles (PAMs), utilizing the highly parallelized **NVIDIA Isaac Lab** environment (Direct Workflow).

---

## 📁 Repository Structure

```text
porcaro_2026/
 ├── scripts/                              # Execution scripts for training and inference
 │   ├── list_envs.py                      # List all registered environments
 │   ├── random_agent.py                   # Random action agent
 │   ├── zero_agent.py                     # Zero action agent
 │   └── rsl_rl/
 │       ├── train.py                      # Main training script
 │       ├── play.py                       # Inference and visualization script
 │       ├── play_sim_midi.py              # MIDI-driven simulation playback
 │       ├── play_sim_rhythm.py            # Rhythm-driven simulation playback
 │       └── cli_args.py                   # Shared CLI argument definitions
 ├── source/
 │   └── porcaro_2026/                     # Core extension package
 │       ├── config/
 │       │   └── extension.toml            # Extension configuration
 │       ├── docs/
 │       │   └── CHANGELOG.rst
 │       └── porcaro_2026/
 │           └── tasks/direct/porcaro_2026/
 │               ├── __init__.py           # Imports user0, user1 submodules
 │               ├── common/              # Shared modules across all users
 │               │   └── actions/
 │               │       ├── base.py
 │               │       ├── pam.py
 │               │       ├── pneumatic.py
 │               │       └── torque.py
 │               ├── user0/               # User 0 implementation
 │               │   ├── __init__.py      # gym.register for user0 tasks
 │               │   ├── porcaro_2026_env.py
 │               │   ├── porcaro_2026_env_cfg.py
 │               │   ├── rhythm_generator.py
 │               │   ├── agents/          # RSL-RL PPO configs
 │               │   │   ├── rsl_rl_ppo_cfg.py
 │               │   │   ├── rsl_rl_ppo_lstm_cfg.py
 │               │   │   └── rsl_rl_ppo_mlp_cfg.py
 │               │   ├── cfg/             # Assets, sensors, controller, rewards params
 │               │   │   ├── actuator_cfg.py
 │               │   │   ├── assets.py
 │               │   │   ├── controller_cfg.py
 │               │   │   ├── logging_cfg.py
 │               │   │   ├── rewards_cfg.py
 │               │   │   └── sensors.py
 │               │   ├── logging/         # Datalogger for Sim-to-Real analysis
 │               │   │   ├── datalogger.py
 │               │   │   └── logging_manager.py
 │               │   └── rewards/
 │               │       └── reward.py
 │               └── user1/               # User 1 implementation (same structure as user0)
 │                   ├── __init__.py      # gym.register for user1 tasks
 │                   └── ...
 ├── environment.yml
 ├── pyproject.toml
 ├── setup.py
 └── README.md
```

---

## 🛠️ Requirements & Installation

We recommend using Miniconda to manage your Python environment.

### 0. Setup Miniconda

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
```

### 1. Create a Conda Environment

```bash
conda create -n porcaro_env python=3.10
conda activate porcaro_env
```

### 2. Install NVIDIA Isaac Lab

This project is built as an extension for NVIDIA Isaac Lab. Please follow the [Official Isaac Lab Installation Guide](https://isaac-sim.github.io/IsaacLab/) and ensure Isaac Lab is installed inside your `porcaro_env` conda environment.

### 3. Install the Porcaro 2026 Package

Clone this repository and install it as an editable Python package:

```bash
git clone https://github.com/actuated-design-lab/porcaro_2026.git
cd porcaro_2026

python -m pip install -e source/porcaro_2026
```

### 4. Verify Installation

After installation, confirm that all task environments are registered correctly:

```bash
python scripts/list_envs.py
```

You should see a table listing all available environments, for example:

```
+-----------------------------------------------+----------------------------------+
| Task Name                                     | Config                           |
+-----------------------------------------------+----------------------------------+
| Template-Porcaro-2026-ModelB-user0            | ...EnvCfg_ModelB                 |
| Template-Porcaro-2026-ModelB-DR-user0         | ...EnvCfg_ModelB_DR              |
| Template-Porcaro-2026-ModelB-user1            | ...EnvCfg_ModelB                 |
| Template-Porcaro-2026-ModelB-DR-user1         | ...EnvCfg_ModelB_DR              |
+-----------------------------------------------+----------------------------------+
```

---

## 🤖 3D Assets & Data Setup (⚠️ CRITICAL)

Due to file size and licensing, 3D models and force map CSVs must be placed manually.

### Robot and Drum Models (`assets/`)

Place the following `.usd` files into `source/porcaro_2026/porcaro_2026/tasks/assets/`:

- `porcaro.usd`
- `sneadrum.usd`
- `drum.usd`

> ⚠️ `sneadrum.usd` is a scaled reference of `drum.usd`. If `drum.usd` is missing, Isaac Sim will not throw an error but the drum will be **invisible** in the GUI. Both files must be present.

### Pneumatic Force Maps (`data/`)

Place the following CSV files into `source/porcaro_2026/porcaro_2026/tasks/data/`:

- `pam_force_map.csv`
- `pam_force_0_map.csv`

---

## 🎮 Usage

Each user has their own independent set of registered task environments. Use the `--task` flag to specify which user's environment to run.

### Available Task IDs

| User  | Task ID (without DR)                      | Task ID (with DR, recommended)               |
|-------|-------------------------------------------|----------------------------------------------|
| user0 | `Template-Porcaro-2026-ModelB-user0`      | `Template-Porcaro-2026-ModelB-DR-user0`      |
| user1 | `Template-Porcaro-2026-ModelB-user1`      | `Template-Porcaro-2026-ModelB-DR-user1`      |

> **DR** (Domain Randomization) is recommended for better sim-to-real transfer.

---

### Training

Train a policy from scratch using your own task environment:

```bash
# user0
python scripts/rsl_rl/train.py --task Template-Porcaro-2026-ModelB-DR-user0

# user1
python scripts/rsl_rl/train.py --task Template-Porcaro-2026-ModelB-DR-user1
```

### Evaluation (Playing)

Watch the trained agent perform in the simulation GUI:

```bash
# user0
python scripts/rsl_rl/play.py --task Template-Porcaro-2026-ModelB-DR-user0

# LSTM
python scripts/rsl_rl/train.py   --task Template-Porcaro-2026-ModelB-DR-user0   --agent rsl_rl_lstm_cfg_entry_point  --seed 1   --lookahead_horizon 0.5 --experiment_name porcaro_lstm_lookahead05   --run_name seed1   --headless

# MLP
python scripts/rsl_rl/train.py   --task Template-Porcaro-2026-ModelB-DR-user0   --agent rsl_rl_mlp_cfg_entry_point  --seed 1   --lookahead_horizon 0.5 --experiment_name porcaro_mlp_lookahead01   --run_name seed1   --headless

# MLP frame stacking
python scripts/rsl_rl/train.py --task Template-Porcaro-2026-ModelB-DR-user0 \
  --agent rsl_rl_mlp_cfg_entry_point \
  --seed 1 \
  --use_frame_stacking --frame_stack_k 5 \
  --lookahead_horizon 0.5 \
  --experiment_name porcaro_mlp_framestack_k5_lookahead05 \
  --run_name seed1 \
  --headless



# user1
python scripts/rsl_rl/play.py --task Template-Porcaro-2026-ModelB-DR-user1
```

---

## 🧩 Adding a New User

To add a new user (e.g., `user2`):

1. Copy the `user0/` directory and rename it to `user2/`.
2. Edit `user2/__init__.py` to register new task IDs (e.g., `Template-Porcaro-2026-ModelB-user2`).
3. Add `from . import user2` to the parent `__init__.py`.
4. Verify with `python scripts/list_envs.py`.