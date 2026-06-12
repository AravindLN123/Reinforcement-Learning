# Project Apex — Code Guide

A navigation reference for every code file in the project. Use the quick-lookup table to find a file by name, or browse the sections below by what you are trying to do.

---

## Quick Lookup

| File | What it does |
|------|-------------|
| `dashboard.py` | Launch the browser-based Control Dashboard (start/stop training, live tuning, snapshots, playback) |
| `playback_best_run.py` | Play back a saved best-lap replay inside TMNF |
| `verify_tminterface.py` | Check that TMInterface 2.1.0 is correctly installed and reachable |
| `requirements-dashboard.txt` | Pip dependencies for the dashboard only |
| `linesight/scripts/train.py` | **Main training entry point** — starts the full IQN training loop |
| `linesight/scripts/launch_game_agade.sh` | Shell script to launch TMNF for training (agade TMInterface build) |
| `linesight/scripts/launch_game_pb.sh` | Shell script to launch TMNF for training (pb TMInterface build) |
| `linesight/config_files/config.py` | **Master config** — all hyperparameters, track selection, reward weights |
| `linesight/config_files/config.default.py` | Read-only defaults; config.py overrides these |
| `linesight/config_files/user_config.py` | Machine-specific paths (TMNF install path, TMInterface path) |
| `linesight/config_files/inputs_list.py` | Defines the 12 discrete keyboard-input combinations (action space) |
| `linesight/config_files/state_normalization.py` | Normalization constants for the 127-dimensional float vector |
| `linesight/trackmania_rl/agents/iqn.py` | **IQN agent** — network architecture (CNN + float head + dueling IQN output) |
| `linesight/trackmania_rl/multiprocess/learner_process.py` | Learner process — pulls batches from buffer, computes IQN loss, updates weights |
| `linesight/trackmania_rl/multiprocess/collector_process.py` | Collector process — runs the game loop, collects experience, pushes to buffer |
| `linesight/trackmania_rl/reward_shaping.py` | Full reward computation (time penalty + distance + VCP shaping + lookahead) |
| `linesight/trackmania_rl/buffer_management.py` | Shared replay buffer management across collector and learner |
| `linesight/trackmania_rl/buffer_utilities.py` | Utilities for sampling and prioritizing experience from the buffer |
| `linesight/trackmania_rl/experience_replay/experience_replay_interface.py` | Interface layer between collector and the replay buffer |
| `linesight/trackmania_rl/tmi_interaction/tminterface2.py` | Low-level TMInterface 2 API wrapper (reads game state, sends inputs) |
| `linesight/trackmania_rl/tmi_interaction/game_instance_manager.py` | Manages the TMNF game process (launch, restart, health checks) |
| `linesight/trackmania_rl/map_loader.py` | Loads `.npy` reference trajectory files for each track |
| `linesight/trackmania_rl/map_reference_times.py` | Stores reference (human) lap times per track for evaluation comparison |
| `linesight/trackmania_rl/geometry.py` | Geometry helpers (VCP projection, distance-along-trajectory, curvature) |
| `linesight/trackmania_rl/contact_materials.py` | Decodes TMNF surface/material codes from game state |
| `linesight/trackmania_rl/analysis_metrics.py` | Computes evaluation metrics (lap time, zone completion, eval_race_time_robust) |
| `linesight/trackmania_rl/utilities.py` | Shared utility functions used across multiple modules |
| `linesight/trackmania_rl/run_to_video.py` | Converts a recorded run into a video file |
| `linesight/trackmania_rl/multiprocess/debug_utils.py` | Debug helpers for the multiprocess setup |
| `tools/export_track_diagrams.py` | Exports top-down track diagrams to `figures/tracks/` |
| `tools/export_training_curves.py` | Exports TensorBoard training curves to `figures/training_curves/` |
| `tools/figures/plot_architecture.py` | Plots the IQN network architecture diagram |
| `tools/figures/plot_comparison.py` | Plots the TMRL vs Linesight comparison figure |
| `tools/figures/plot_reward_components.py` | Plots the reward component breakdown figure |
| `tools/figures/plot_schedules.py` | Plots hyperparameter schedule curves (epsilon, lr) |
| `tools/figures/plot_training_curves.py` | Plots training curves from TensorBoard event files |
| `tools/figures/plot_trajectory.py` | Plots agent trajectory overlaid on the track centerline |
| `linesight/scripts/tools/gbx_to_vcp.py` | Converts a `.Replay.Gbx` centerline recording into a `.npy` VCP trajectory file |
| `linesight/scripts/tools/gbx_to_times_list.py` | Extracts lap times from a `.Replay.Gbx` file |
| `linesight/scripts/tools/tmi2/add_vcp_as_triggers.py` | Adds VCP triggers to a map file (used when setting up a new track) |
| `linesight/scripts/tools/tmi2/add_cp_as_triggers.py` | Adds checkpoint triggers to a map file |
| `linesight/scripts/tools/video_stuff/inputs_to_gbx.py` | Converts a saved `.inputs` file back to a `.Replay.Gbx` replay |
| `linesight/scripts/tools/video_stuff/animate_race_time.py` | Generates an animated race-time overlay for video exports |

---

## By Task

### I want to start training

1. Open `linesight/config_files/user_config.py` and make sure the TMNF and TMInterface paths are correct for your machine.
2. Open `linesight/config_files/config.py` and set the track name and any hyperparameters you want to change.
3. Launch TMNF using one of the shell scripts in `linesight/scripts/`:
   - `launch_game_agade.sh` or `launch_game_pb.sh` (depending on your TMInterface build).
4. Run `linesight/scripts/train.py` to start training.

Or use the dashboard (see below) to do steps 3-4 from the browser.

---

### I want to use the Control Dashboard

```
python dashboard.py
```

Opens a Streamlit page in your browser. From there you can start/stop training, adjust hyperparameters live, trigger weight snapshots, and play back recorded laps. See `docs/DASHBOARD.md` for the full feature list.

---

### I want to change a hyperparameter

Edit `linesight/config_files/config.py`. The most commonly tuned settings:

| Setting | What it controls |
|---------|-----------------|
| `global_schedule_speed` | How fast epsilon and learning rate decay |
| `memory_size` | Replay buffer size |
| `batch_size` | Training batch size |
| `learning_rate` | Initial learning rate |
| `tm_engine_step_per_action` | How many game ticks per agent action step |
| `reward_*` weights | Individual reward component weights |

If training is running, the dashboard can hot-reload changes to config.py without restarting.

---

### I want to add a new track

Follow `docs/ADD_NEW_TRACK.md`. The key scripts involved:

1. Record a centerline lap in TMNF and save the replay.
2. Run `linesight/scripts/tools/gbx_to_vcp.py` to convert the replay to a `.npy` trajectory file.
3. Place the `.npy` file in `linesight/maps/`.
4. Update the track name in `linesight/config_files/config.py`.

---

### I want to play back the best recorded lap

```
python playback_best_run.py
```

Plays back the best saved replay inside a running TMNF instance. Replays are stored in `results/`.

---

### I want to export figures for the report

Run any script inside `tools/` or `tools/figures/`. Output goes directly to `report/Images/`:

| Subfolder | Contents |
|-----------|----------|
| `report/Images/tracks/` | Top-down track diagrams |
| `report/Images/training_curves/` | Training curve plots |
| `report/Images/methods/` | Architecture and reward diagrams |
| `report/Images/dashboard/` | Dashboard screenshots |
| `report/Images/appendix/` | Figures for the previous-approaches appendix |

---

### I want to understand the IQN agent

Start with these files in order:

1. `linesight/trackmania_rl/agents/iqn.py` — the network (CNN front-end, float vector branch, IQN head with dueling output)
2. `linesight/trackmania_rl/multiprocess/learner_process.py` — how the network is trained (loss computation, optimizer step)
3. `linesight/trackmania_rl/multiprocess/collector_process.py` — how experience is collected from the game
4. `linesight/trackmania_rl/reward_shaping.py` — how the reward is computed at each step

Then read `docs/LINESIGHT_EXPLAINED.md` for the full technical walkthrough.

---

### I want to understand the observation space

- **Image stream:** 160x120 greyscale frames captured from TMNF — processed inside `collector_process.py`.
- **Float vector (127 dims):** defined and normalized in `linesight/config_files/state_normalization.py`; the values come from the game state read by `tmi_interaction/tminterface2.py`.
- **VCP positions:** loaded from `.npy` files by `map_loader.py` and projected onto the car's position by `geometry.py`.

---

## Data Files

### Track map files (`maps/`)

| File | Description |
|------|-------------|
| `ovaltrack1.Challenge.Gbx` | TMNF map file for the oval proof-of-concept track |
| `Figure8Track.Challenge.Gbx` | TMNF map file for the figure-eight demo track |
| `Monza.Challenge.Gbx` | TMNF map file for the Monza-in-TMNF track |
| `ovaltrack1_0.5m_cl.npy` | Reference centerline trajectory for ovaltrack1 |
| `*_centerline.Replay.Gbx` | Human-driven centerline lap recordings (source for `.npy` files) |

### Reference trajectories (`linesight/maps/`)

Pre-converted `.npy` trajectory arrays used during training. Each file stores waypoints sampled every 0.5 m along the track centerline.

| File | Track |
|------|-------|
| `ovaltrack1_0.5m_cl.npy` | ovaltrack1 |
| `Figure8Track_0.5m_cl.npy` | Figure8Track |
| `Monza_0.5m_cl.npy` | Monza-in-TMNF |
| `map5_0.5m_cl.npy` | Additional map (not a primary target) |
| `ESL-Hockolicious_0.5m_cl2.npy` | Additional map (not a primary target) |

### Best-lap replays (`results/`)

| File | Description |
|------|-------------|
| `Figure8Track_linesight_best_43s56.Replay.Gbx` | Best recorded lap on Figure8Track (43.56 s) |
| `ovaltrack1_linesight_centerline_ref.Replay.Gbx` | Best recorded lap on ovaltrack1 (30.9 s) |

### Training logs (`linesight/tensorboard/`)

TensorBoard event files from completed training runs. Open with:

```
tensorboard --logdir linesight/tensorboard/
```

| Subfolder | Run |
|-----------|-----|
| `ovaltrack1_run01/` | First ovaltrack1 training run |
| `figure8_run01/` | Figure8Track + Monza shared-buffer run |
| `figure8_run01_2/` | Continuation of figure8 run |
| `lay_mono/` | Monza-specific training segment |

---

## Key Directories at a Glance

```
Code/
├── dashboard.py              # Control Dashboard entry point
├── playback_best_run.py      # Replay playback entry point
├── verify_tminterface.py     # TMInterface health check
├── docs/                     # All user-facing guides (start here)
├── figures/                  # Generated figures for the report
├── maps/                     # TMNF map files + one reference trajectory
├── results/                  # Best-lap replay files
├── tools/                    # Figure export scripts
└── linesight/
    ├── config_files/         # All configuration (edit these to tune)
    ├── scripts/
    │   ├── train.py          # Training entry point
    │   ├── launch_game_*.sh  # Game launch scripts
    │   └── tools/            # Track setup and replay utilities
    ├── trackmania_rl/
    │   ├── agents/iqn.py     # IQN network definition
    │   ├── multiprocess/     # Learner + collector processes
    │   ├── tmi_interaction/  # TMInterface API wrapper
    │   └── ...               # Reward, geometry, buffer, utilities
    ├── maps/                 # Reference trajectory .npy files
    └── tensorboard/          # Training logs (open with TensorBoard)
```
