<p align="center">
  <img src="https://github.com/AravindLN123/Reinforcement-Learning/blob/main/ProjectApex_Logo_Big.png" alt="Project Apex Logo" width="1000"/>
</p>

# Project Apex — Software

**Project Apex** trains an autonomous race driver on Trackmania Nations Forever (TMNF) using the Linesight framework (by the Linesight-RL team) and Implicit Quantile Networks (IQN). This folder contains everything needed to run training, tune the agent live from a browser, and play back recorded laps.

## Quick navigation

| What you want to do | Read |
|---|---|
| Install TMNF, TMInterface, Python, and all dependencies | [docs/INSTALL.md](docs/INSTALL.md) |
| Day-by-day training workflow | [docs/QUICKSTART.md](docs/QUICKSTART.md) |
| Drive training from a browser (start/stop, live hyperparameter tuning, snapshots, replay) | [docs/DASHBOARD.md](docs/DASHBOARD.md) |
| Teach the agent a new track | [docs/ADD_NEW_TRACK.md](docs/ADD_NEW_TRACK.md) |
| Understand the IQN agent internals | [linesight/README.md](linesight/README.md) |

## Folder map

```
Code/
├── README.md                      <- you are here
├── dashboard.py                   <- Project Apex Control Dashboard (Streamlit)
├── playback_best_run.py           <- replay any recorded lap inside TMNF
├── verify_tminterface.py          <- sanity-check that TMInterface is running
├── requirements-dashboard.txt     <- pip deps for the dashboard only
│
├── docs/
│   ├── INSTALL.md                 <- full environment setup
│   ├── QUICKSTART.md              <- day-by-day workflow cheat sheet
│   ├── DASHBOARD.md               <- Control Dashboard user guide
│   ├── ADD_NEW_TRACK.md           <- adding a new Trackmania track
│   ├── CODE_GUIDE.md              <- full file reference + task-based navigation
│   └── LINESIGHT_EXPLAINED.md     <- deep technical reference for the IQN agent
│
├── linesight/                     <- Linesight framework (Linesight-RL team, flattened copy)
│   ├── config_files/              <- config.py, user_config.py, state normalization
│   ├── scripts/                   <- train.py and launch shell scripts
│   ├── trackmania_rl/             <- IQN agent core (18 Python files)
│   ├── maps/                      <- reference trajectory .npy files
│   ├── tensorboard/               <- training event logs (inspectable without weights)
│   └── save/                      <- trained model weights (NOT in repo -- see below)
│
├── maps/                          <- TMNF map files (.Gbx) + reference centerlines (.npy)
├── results/                       <- best-lap replays (.Replay.Gbx)
└── tools/                         <- figure and metrics export scripts
```

## Trained Model Weights

The `linesight/save/` folder (~1.2 GB of `.torch` checkpoint files) is **not stored in this repository** to keep the repo size manageable.

To use a pre-trained model:

- **Download from Google Drive:** [https://drive.google.com/drive/folders/1ad6HIPwomb4y6Zn\_nF2eptqvb9F4lq4\_?usp=sharing](https://drive.google.com/drive/folders/1ad6HIPwomb4y6Zn_nF2eptqvb9F4lq4_?usp=sharing)
  Place the downloaded `save/` folder at `Code/linesight/save/`.

- **Retrain from scratch:** follow [docs/QUICKSTART.md](docs/QUICKSTART.md). Expect roughly 4-24 hours per track on a modern NVIDIA GPU.

The TensorBoard event files under `linesight/tensorboard/` and the best-lap replays under `results/` ARE committed, so the training history and end results are inspectable without the weights.

## Tech stack

| Component | Detail |
|---|---|
| Language | Python 3.11 |
| RL framework | Linesight by the Linesight-RL team (IQN / distributional DQN) |
| Game | Trackmania Nations Forever via TMInterface 2.1.0 and TMLoader |
| Deep learning | PyTorch with CUDA |
| Operator UI | Streamlit (Project Apex Control Dashboard) |
| Monitoring | TensorBoard |

## Three tracks trained

| Track | Type | Best lap |
|---|---|---|
| `ovaltrack1` | Simple oval (proof of concept) | 30.9 s |
| `Figure8Track` | Figure-8 (primary demo) | 43.56 s |
| `Monza-in-TMNF` | Grand-Prix circuit (hardest) | 1:27.96 |

All three were trained in a single run with a shared replay buffer (transfer learning across tracks).
