Project Apex

<p align="center">
  <img src="Code/assets/ProjectApex_Logo_Big.png" alt="Project Apex Logo" width="1000"/>
</p>

## Project Overview

**Project Apex** is an autonomous-racing project built on the open-source Linesight framework (by the Linesight-RL team) a distributional deep Reinforcement Learning (RL) approach for Trackmania Nations Forever (TMNF). The trained agent drives the in-game car around custom Trackmania tracks using a vision-based observation stack (160 × 120 greyscale image plus a 127-dimensional state vector) and a discrete action space, all powered by Implicit Quantile Networks (IQN).

The project arrived at this approach after iterating through two earlier directions (F1Tenth with Proximal Policy Optimization, then TMRL with Soft Actor-Critic on Trackmania 2020). Project Apex on the Linesight framework is the chosen final approach; the earlier attempts are documented in the Appendix of the report as part of the iterative exploration narrative.

## Repository

GitHub repository: [Wings-hub/BA26-Project-B](https://github.com/Wings-hub/BA26-Project-B.git)

## Core Idea

- **Framework:** Linesight by the Linesight-RL team (used as the underlying agent stack, credited in the report and in code comments)
- **Game:** Trackmania Nations Forever (TMNF) via the TMInterface 2.1.0 plugin and TMLoader
- **Tracks:** `ovaltrack1` (proof of concept), `Figure8Track` (primary demo), `Monza-in-TMNF` (most complex)
- **Input:** 160 × 120 greyscale image fed through a Convolutional Neural Network (CNN) + 127-dimensional float vector (car state, velocities, upcoming Virtual Checkpoint Points)
- **Action space:** Discrete — 12 combinations of keyboard arrow keys
- **Algorithm:** Implicit Quantile Networks (IQN) with dueling heads — a distributional Deep Q-Network variant
- **Reward design:** Time penalty + per-meter distance reward + potential-based shaping toward upcoming checkpoints, with a 7-second lookahead window that discourages slow wall-hugging
- **Scope:** Single framework (Linesight), one game (TMNF), three tracks with shared replay buffer (transfer learning)

## Problem Statement

Training an RL agent to race fast and cleanly on a Trackmania circuit from raw visual input is challenging: the policy must interpret high-dimensional pixel data, learn long-horizon racing behaviour (braking, racing line, recovery) from delayed reward signals, and avoid the classic failure mode of wall-hugging — where the agent learns to scrape walls for slow but reliable progress instead of carving a clean racing line.

Project Apex addresses this by combining Linesight's distributional IQN agent (which estimates the full distribution of returns rather than just the mean), a CNN-based vision stack, and a 7-second lookahead reward that implicitly penalises slow wall-contact driving.

## Objectives

- Apply the Linesight framework to a chosen set of Trackmania Nations Forever tracks and reach a clean racing line (no wall-hugging)
- Achieve ≥ 80% lap completion across the three target tracks
- Compare the final approach against the previous attempts (F1Tenth/PPO and TMRL/SAC) as documented in the Appendix
- Provide an accessible operator experience through the **Project Apex Control Dashboard** — a Streamlit web app that lets non-technical users drive training, tune hyperparameters live, snapshot weights, and play back recorded laps
- Position the project as a **playground for future student work**, with concrete extension ideas listed in the report's Next Steps section
- Document the project as a structured LaTeX report following the Knowledge Discovery in Databases (KDD) methodology

## MDP Formulation

The racing task is framed as a Markov Decision Process (MDP):

- **State space:** 160 × 120 greyscale image (CNN-processed) plus a 127-dimensional float vector containing car physics, velocity, upcoming Virtual Checkpoint Points (VCPs) along the reference trajectory, and the previous four actions
- **Action space:** Discrete — 12 keyboard combinations (arrow keys)
- **Reward design:**
  - Per-step time penalty
  - Per-meter distance reward along the reference trajectory
  - Potential-based shaping toward upcoming VCPs (does not bias the optimal policy)
  - Implicit lookahead: a 7-second progress window makes slow wall-contact strictly inferior to fast clean driving

This setup encourages forward progress along a recorded reference trajectory while discouraging the wall-hugging local minimum that limited earlier attempts.

## Methodology

The project follows a Knowledge Discovery in Databases (KDD)-style workflow adapted to RL:

1. **Selection.** Choose the framework (Linesight) and game (TMNF), record reference trajectories, design tracks.
2. **Preprocessing.** Image conversion (BGRA → greyscale, normalisation) and float-vector normalisation; reward-component computation per step.
3. **Transformation.** Frame the racing task as an MDP with a discrete action space and a multi-component reward including lookahead.
4. **Mining.** Train the IQN agent via Linesight's multiprocess collector + learner architecture; tune hyperparameter schedules; apply transfer learning across the three tracks.
5. **Interpretation.** Evaluate the policy on each track with recorded replays; compare against the prior approaches (F1Tenth/PPO, TMRL/SAC) and analyse the racing-line quality.

## Evaluation Metrics

- **Best lap time per track** — primary headline metric
- **Lap completion rate** — fraction of evaluation episodes that finish a clean lap (target ≥ 80%)
- **Racing-line quality** — visual inspection of recorded replays; whether the agent drives a clean line or hugs walls
- **TensorBoard scalars** — `avg_Q`, `single_zones_reached`, IQN loss, `mean_race_time`, `eval_race_time_robust`
- **Robustness** — behaviour across the three tracks with shared replay buffer
- **Iteration count to convergence** — for the comparative analysis with prior attempts

## Project Journey (Iterative Exploration)

Project Apex was the third RL approach attempted. Both earlier attempts are documented in the Appendix of the report:

- **F1Tenth with PPO on Monza** — initial direction; partial training data
- **TMRL with SAC on Trackmania 2020** — 13 iterations across three days; converged to wall-hugging at ~26 km/h, never recovered to a clean racing line
- **Project Apex with the Linesight framework / IQN on Trackmania Nations Forever** — chosen final approach; clean racing line on all three target tracks

## Delighters (Extra-mark Highlights)

1. **Iterative exploration** — the three-approach journey is documented as a learning story, not hidden
2. **Project Apex Control Dashboard** — Streamlit web app for non-technical users (start/stop training, live hyperparameter tuning, weight snapshots, replay playback)
3. **Playground for future students** — the project is intentionally positioned as a re-usable platform; concrete extension ideas (frame stacking, saliency analysis, transformer-based vision, multi-track generalisation, and more) are catalogued in the Next Steps section of the Conclusion

## Repository Structure

```text
01-Reinforcement/
|-- README.md                          # this file (living document)
|-- .gitattributes                     # Git LFS rules (future-proofing for large binaries)
|-- .gitignore                         # LaTeX auxiliary file exclusions
|-- author.xlsx                        # Author / project metadata
|
|-- Code/                              # Project Apex software (see Code/README.md for details)
|   |-- .gitignore                     # excludes trained-weight artifacts, caches, venvs
|   |-- README.md                      # software-specific overview
|   |-- dashboard.py                   # Project Apex Control Dashboard (Streamlit)
|   |-- playback_best_run.py
|   |-- verify_tminterface.py
|   |-- requirements-dashboard.txt
|   |-- docs/                          # all user-facing guides
|   |   |-- INSTALL.md                 # detailed environment setup
|   |   |-- QUICKSTART.md              # day-by-day workflow
|   |   |-- LINESIGHT_EXPLAINED.md     # deep technical reference for the IQN agent
|   |   |-- DASHBOARD.md               # Project Apex Control Dashboard guide
|   |   |-- CODE_GUIDE.md              # full code file reference and task-based navigation
|   |   `-- ADD_NEW_TRACK.md           # adding a new Trackmania track
|   |-- tools/                         # figure / metrics export scripts (output → report/Images/)
|   |-- maps/                          # TMNF map files (.Gbx) + reference centerlines (.npy)
|   |-- results/                       # best-known recorded replays (.Replay.Gbx)
|   `-- linesight/                     # flattened Linesight framework source + docs + tensorboard logs
|
|-- report/                            # LaTeX thesis report
|   |-- Contents/General/              # chapter content (LaTeX source)
|   |-- General/                       # shared LaTeX configuration and assets (template — do not modify)
|   |-- Images/                        # all project figures (tracks, training curves, methods, dashboard, appendix)
|   |-- System/EdgeComputer/           # main LaTeX entry point and compiled PDF
|   `-- tikz/                          # TikZ components
|
|-- MLbib/                             # bibliography sources (BibTeX)
|
|-- Manual/                            # (scaffold) hardware/user manual placeholder
|-- Poster/                            # (scaffold) academic poster
|-- Presentations/                     # (scaffold) presentation templates
`-- ProjectManagement/                 # (scaffold) project-management documents
```

## Setup and Quick Start

For installing and running the Project Apex software, follow the documentation under [Code/](Code/):

1. **Navigate the codebase** — see [Code/docs/CODE_GUIDE.md](Code/docs/CODE_GUIDE.md) for a full file-by-file reference with quick-lookup table and task-based navigation
2. **Install dependencies** — see [Code/docs/INSTALL.md](Code/docs/INSTALL.md) (Python 3.11, PyTorch + CUDA, TMNF, TMInterface 2.1.0, TMLoader, the Linesight framework dependencies, Streamlit for the dashboard)
3. **Run the day-to-day workflow** — see [Code/docs/QUICKSTART.md](Code/docs/QUICKSTART.md)
4. **Drive the project from a browser** — see [Code/docs/DASHBOARD.md](Code/docs/DASHBOARD.md) for the Project Apex Control Dashboard
5. **Add a new track** — see [Code/docs/ADD_NEW_TRACK.md](Code/docs/ADD_NEW_TRACK.md)

## Trained Model Weights

Trained model checkpoints (~1.2 GB of `.torch` files under `Code/linesight/save/`) are **not stored in this repository** to keep the repo size manageable.

To use a pre-trained model, choose one of:

- **Download from Google Drive:** [https://drive.google.com/drive/folders/1ad6HIPwomb4y6Zn\_nF2eptqvb9F4lq4\_?usp=sharing](https://drive.google.com/drive/folders/1ad6HIPwomb4y6Zn_nF2eptqvb9F4lq4_?usp=sharing)
  Place the downloaded `save/` folder at `Code/linesight/save/`.
- **Retrain from scratch** by following [Code/docs/QUICKSTART.md](Code/docs/QUICKSTART.md) — expect roughly 4–24 hours per track on a modern NVIDIA GPU

The recorded TensorBoard event files under `Code/linesight/tensorboard/` and the best-lap replays under `Code/results/` ARE in the repository, so the training history and end results are inspectable without the weights themselves.

## Note on Git LFS

The repository includes a scoped `.gitattributes` at `01-Reinforcement/.gitattributes` that routes `.torch`, `.pkl`, `.h5`, `.ckpt`, `.mp4`, and `.mov` files through Git LFS. This is **future-proofing** — at the time of writing, no committed file matches these patterns, so cloning the repository does NOT require Git LFS to be installed.

If anyone later commits a file matching these patterns (for example, a small `.torch` artifact that needs to be shared), LFS routing kicks in automatically. The one-time setup is:

```bash
brew install git-lfs        # macOS, or use the appropriate package manager
git lfs install             # one-time per user account
```

## Build the Report

From the repository root:

```bash
cd 01-Reinforcement/report/System/EdgeComputer
pdflatex ProjectApex.tex
biber ProjectApex
pdflatex ProjectApex.tex
pdflatex ProjectApex.tex
```

If you use `latexmk`, an equivalent automated workflow is also fine.

## Tech Stack

- **Language:** Python 3.11
- **RL framework:** the Linesight framework by the Linesight-RL team (used as the underlying agent stack)
- **Game / Simulator:** Trackmania Nations Forever (TMNF) via TMInterface 2.1.0 and TMLoader
- **Deep-learning framework:** PyTorch with CUDA
- **Operator UI:** Streamlit (Project Apex Control Dashboard)
- **Monitoring:** TensorBoard
- **Documentation:** LaTeX, BibLaTeX
- **Version control:** Git and GitHub

## Authors

- Karrar Al-Ameeri
- Aravind Lakshmi Narayanan
