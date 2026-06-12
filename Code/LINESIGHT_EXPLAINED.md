# Linesight — Full Technical Explanation

This document is the authoritative technical reference for how Linesight works in this project. It is written for someone who understands Python and machine learning basics but has not read the Linesight source code. All file paths are relative to `linesight/` inside this repo.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  TMNF Game (runs via TMLoader + TMInterface plugin)                  │
│                                                                      │
│  Game loop at 100 Hz  →  TMInterface exposes game state via socket  │
└──────────────────────────────┬───────────────────────────────────────┘
                               │  TCP socket (port 8478)
                               │  Sends: frame + float state
                               │  Receives: keyboard action
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  game_instance_manager.py  (collector process)                       │
│                                                                      │
│  1. requests 160×120 BGRA frame from TMInterface                    │
│  2. converts to grayscale  → shape (1, 120, 160) uint8              │
│  3. reads 127-dim float state vector from TMInterface               │
│  4. passes (frame, floats) to IQN network for action selection      │
│  5. sends selected action back via socket                           │
│  6. stores (state, action, reward, next_state) in replay buffer     │
└──────────────────────────────┬───────────────────────────────────────┘
                               │  shared memory / multiprocessing queues
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  learner_process.py  (learner process)                               │
│                                                                      │
│  1. samples batch of 512 transitions from replay buffer             │
│  2. computes IQN quantile loss                                      │
│  3. gradient step on online network                                 │
│  4. soft-updates target network (τ = 0.02)                          │
│  5. logs to TensorBoard                                             │
└─────────────────────────────────────────────────────────────────────┘
```

Linesight runs two OS processes simultaneously:
- **Collector**: plays the game, collects experience
- **Learner**: reads experience from the buffer, trains the network

They communicate via shared memory queues. The collector always runs at real-time game speed; the learner runs as fast as the GPU allows.

---

## Layer 1 — Observation Pipeline

**Relevant files:**
- `trackmania_rl/tmi_interaction/game_instance_manager.py` — capture + preprocessing
- `trackmania_rl/tmi_interaction/tminterface2.py` — socket protocol
- `config_files/state_normalization.py` — float normalization constants
- `config_files/config.py` — resolution settings

### 1a. Screen capture

Every action step (every 50 ms of game time), the collector:
1. Sends a `C_REQUEST_FRAME` message via the TMInterface socket specifying width=160, height=120
2. TMInterface renders the game at that resolution and sends back 160×120×4 bytes (BGRA format)
3. The collector calls `cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)` → `(120, 160)` uint8
4. Adds a channel dimension → `(1, 120, 160)` uint8

**No frame stacking.** Only the current frame is used. Temporal information (how fast the car is moving, which direction it's turning) is NOT derived from multiple images — it comes from the float vector instead. This is a design choice that keeps the CNN simple but means the network cannot use optical flow or motion blur from images.

**Normalization at inference time** (in `iqn.py:infer_network()`):
```python
img_tensor = (torch.from_numpy(frame).float() - 128) / 128
# Maps uint8 [0, 255] → float32 [-1.0, 1.0]
```

### 1b. Float state vector (127 dimensions)

In addition to the image, every step the collector reads a structured float vector from TMInterface. This encodes everything the agent needs to know about its physical state and upcoming track geometry. The vector is normalized using precomputed per-feature mean and standard deviation from `config_files/state_normalization.py`.

| Indices | Meaning | Notes |
|---------|---------|-------|
| `[0]` | Time remaining in mini-race (normalized) | Fraction of 7-second horizon left |
| `[1–20]` | 4 × upcoming checkpoint direction + distance | 5 floats per checkpoint (unit vector to next 4 VCPs) |
| `[21–36]` | Previous 4 actions (one-hot, 4 bits each) | Gives the agent memory of what it just did |
| `[37–56]` | Car physics | Gear, wheels on ground, steering angle, etc. |
| `[57–62]` | Velocity (3D) + angular velocity (3D) | Speed and rotation in world frame |
| `[63–182]` | 40 upcoming Virtual Checkpoint Points (VCPs) × 3 coords | Lookahead "tunnel" the agent sees ahead |
| `[183]` | Distance to finish line | Normalized |
| `[184]` | Is-freewheeling flag | 1.0 if engine cut, 0.0 otherwise |

**Why floats AND image?** The image tells the agent what it sees (track surface, walls, sky). The float vector tells it where it is and what's coming. The CNN can't reliably read speed from a single frame; the float vector supplies that directly.

### 1c. Reference trajectory (Virtual Checkpoint Points)

Before training, you record a centerline lap and convert it to a `.npy` file of 3D coordinates, spaced 0.5 m apart. These are the **Virtual Checkpoint Points (VCPs)**. During training:
- The game reports how far the car has progressed along this list
- 40 upcoming VCP coordinates are always included in the float vector (the "track tunnel" ahead)
- The reward is computed based on distance advanced along this list

The reference doesn't need to be fast — it just needs to be clean and continuous.

---

## Layer 2 — Neural Network Architecture

**File:** `trackmania_rl/agents/iqn.py` — class `IQN_Network`

### Why IQN instead of plain DQN?

Standard DQN learns `Q(s, a) = E[total return]` — just the **mean** expected return. IQN (Implicit Quantile Networks) learns the full **distribution** of returns, represented as a set of quantile estimates `Q_τ(s, a)` for τ ∈ [0, 1]. 

Why this matters for racing:
- Racing has high variance (one mistake crashes everything). The mean return doesn't capture this.
- IQN's distributional loss uses more information per training step → faster convergence
- Distributional Q-values are more stable under sudden reward changes (like hitting a wall)

### Full architecture diagram

```
Image input: (1, 120, 160) uint8  →  normalized float32
                    │
            ┌───────▼────────┐
            │   CNN Head      │
            │ Conv2d(1→16)    │  kernel 4×4, stride 2  →  (16, 59, 79)
            │ LeakyReLU       │
            │ Conv2d(16→32)   │  kernel 4×4, stride 2  →  (32, 28, 38)
            │ LeakyReLU       │
            │ Conv2d(32→64)   │  kernel 3×3, stride 2  →  (64, 13, 18)
            │ LeakyReLU       │
            │ Conv2d(64→32)   │  kernel 3×3, stride 1  →  (32, 11, 16)
            │ LeakyReLU       │
            │ Flatten         │  →  5632 dims
            └───────┬─────────┘
                    │
Float input: (127,) float32  →  normalized
                    │
            ┌───────▼─────────┐
            │  Float MLP Head  │
            │ Linear(127→256)  │
            │ LeakyReLU        │
            │ Linear(256→256)  │
            │ LeakyReLU        │
            └───────┬──────────┘
                    │ 256 dims
                    │
            ┌───────▼──────────────────────────────────┐
            │          Concatenate: 5632 + 256 = 5888  │
            └───────┬──────────────────────────────────┘
                    │
                    │    ┌──────────────────────────────────┐
                    │    │  IQN Quantile Embedding           │
                    │    │  τ ~ Uniform(0,1), shape (N,)    │
                    │    │  cosine basis: cos(i·π·τ), i=1..64│
                    │    │  Linear(64 → 5888)               │
                    │    │  LeakyReLU                       │
                    │    └──────────────┬───────────────────┘
                    │                   │  (N, 5888)
                    └──────────────────►│
                           Hadamard product (element-wise ×)
                                        │
                              (N × batch, 5888)
                                        │
              ┌─────────────────────────┤
              ▼                         ▼
     ┌──────────────────┐    ┌──────────────────┐
     │  Advantage Head  │    │   Value Head      │
     │ Linear(5888→512) │    │ Linear(5888→512)  │
     │ LeakyReLU        │    │ LeakyReLU         │
     │ Linear(512→12)   │    │ Linear(512→1)     │
     │   A(s,a,τ)       │    │   V(s,τ)          │
     └────────┬─────────┘    └────────┬──────────┘
              │                       │
              └──────────┬────────────┘
                Q(s,a,τ) = V(s,τ) + A(s,a,τ) - mean_a[A(s,a,τ)]
                         (N × batch, 12 actions)
```

### Dueling architecture explained

Separating V (how good is this state?) from A (how much better is this action than average?) stabilizes learning. If the car is on a straight with no obstacles, all 12 actions have similar Q-values — dueling captures this by making V large and A near-zero. Without dueling, each action's Q-value would need to independently learn that "going straight here is generally good."

### IQN loss (quantile Huber / pinball loss)

```python
TD_error = target_Q_τ' - predicted_Q_τ    # (batch, N_target, N_pred)
huber = where(|TD_error| < κ, 0.5·TD_error²/κ, |TD_error| - 0.5·κ)
# κ = iqn_kappa = 5e-3

loss = (|τ - 𝟙[TD_error < 0]| · huber).mean()
# The asymmetric weighting is what makes it a quantile estimator:
# τ = 0.1 means "predict the 10th percentile of returns"
```

Training uses `iqn_n = 8` quantile samples per step; inference uses `iqn_k = 32` (average of 32 Q-value estimates) to get a stable action selection.

---

## Layer 3 — Reward Function

**File:** `trackmania_rl/buffer_management.py`

### Reward components

Every transition (50 ms of game time) accumulates:

```
reward(t) = time_penalty + distance_reward + potential_shaping_delta + engineered_bonuses
```

#### 1. Time penalty
```
time_penalty = constant_reward_per_ms × ms_per_action
             = (-6/5000) × 50
             = -0.06 per action step
```
Forces the agent to minimize time. Without this, the agent could learn to drive infinitely slowly and still accumulate distance rewards.

#### 2. Distance reward
```
distance_reward = reward_per_m_advanced × (meters_at_t - meters_at_{t-1})
                = (5/500) × Δmeters
                = 0.01 per meter advanced along reference trajectory
```
The reference trajectory is the `.npy` centerline file. Distance is measured as progress along this line (not Euclidean distance to a point). Wall-scraping still advances distance, but more slowly than clean driving — combined with the time penalty, the net reward for wall-hugging is lower than for clean racing.

#### 3. Potential-based reward shaping (Ng et al. 1999)
```python
def get_potential(state_float):
    dist_to_next_vcp = np.linalg.norm(state_float[62:65])  # 3D distance to next checkpoint
    dist_clamped = clamp(dist_to_next_vcp, min_dist, max_dist)
    return shaped_reward_dist_to_cur_vcp × dist_clamped
```
The reward delta is `γ·Φ(s') - Φ(s)`. This adds a dense signal encouraging the agent to get closer to the next VCP each step, without biasing the optimal policy (proven by Ng et al.). Think of it as a local "GPS pull" toward the next checkpoint.

#### 4. Engineered bonuses (all 0 by default)
These are professional TMNF technique rewards that are dormant in our training:
- `engineered_speedslide_reward` — rewards speedsliding (a TMNF-specific technique where you slide sideways for speed)
- `engineered_neoslide_reward` — rewards neosliding (another advanced slide technique)
- `engineered_kamikaze_reward` — rewards high-speed cornering ("kamikaze" turns)
- `engineered_close_to_vcp_reward` — extra reward for being precisely on the reference line

**Do not activate these** unless the agent has already learned clean laps and you want to push it toward professional-level techniques. Activating them prematurely creates reward hacking.

### Multi-step returns (n-step = 3)

Rather than bootstrapping off the next state alone (1-step TD), Linesight accumulates reward over 3 steps before bootstrapping:

```
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + γ³·Q(s_{t+3}, argmax_a Q(s_{t+3}, a))
```

This reduces bootstrapping bias early in training (the target network's Q estimates are poor early on, so reducing reliance on them by taking 3 real reward steps helps). The gamma schedule transitions to `γ = 1.0` at 2.5M steps, meaning the agent eventually cares about infinite-horizon returns.

### Why this reward fixes Phase 1's wall-hugging problem

Phase 1 used a per-step centerline progress reward. Scraping a wall STILL advances along the centerline — just slowly. Low speed + low variance = wall-hugging was genuinely reward-optimal.

Linesight's time penalty kills the low-speed advantage: driving at 20 km/h accumulates the same `-0.06` time penalty per step as driving at 200 km/h, but only a fraction of the distance reward. Fast clean driving is now strictly better than slow wall-hugging in expected return.

---

## Layer 4 — Hyperparameters

**File:** `config_files/config.py`

All time-varying parameters use "schedules": a list of `(step, value)` pairs where Linesight linearly interpolates between breakpoints.

### Learning rate schedule
```python
lr_schedule = [
    (0,           1e-3),    # Start high for fast early exploration
    (3_000_000,   5e-5),    # Drop after 3M steps (model stabilizing)
    (12_000_000,  5e-5),    # Hold
    (15_000_000,  1e-5),    # Final fine-tune
]
```
The `global_schedule_speed` multiplier compresses or stretches all schedules. Default is 1.0; for a simple track like `apex-bone`, setting `global_schedule_speed = 0.8` makes all milestones hit ~20% earlier.

### Exploration schedule (ε-greedy)
```python
epsilon_schedule = [
    (0,         1.0),   # 100% random exploration at start
    (50_000,    1.0),   # Stay random while buffer fills
    (300_000,   0.1),   # Drop to 10% after initial fills
    (3_000_000, 0.03),  # 3% random for final polish
]
```
Also uses Boltzmann noise (`epsilon_boltzmann_schedule`, τ=0.01) as additive exploration on top of ε-greedy.

### Memory / replay buffer schedule
```python
memory_size_schedule = [
    (0,           (50_000,  20_000)),    # (train_buffer, test_buffer)
    (5_000_000,   (100_000, 75_000)),
    (7_000_000,   (200_000, 150_000)),
]
```
The buffer grows over training to balance sample diversity. Too small early = catastrophic forgetting; too large = slow sampling from old stale transitions.

### Gamma (discount) schedule
```python
gamma_schedule = [
    (0,          0.999),   # Near-sighted early (bootstrap soon)
    (1_500_000,  0.999),   # Hold
    (2_500_000,  1.0),     # Full undiscounted at 2.5M steps
]
```
Starting with γ < 1 prevents the early target values from exploding (since Q estimates are noisy early). Transitioning to γ=1 at 2.5M aligns with the n-step returns becoming the dominant training signal.

### Optimizer settings
```python
batch_size     = 512        # Large batches stabilize IQN loss
weight_decay   = lr / 50    # Tied to LR for implicit regularization
adam_epsilon   = 1e-4       # Larger than PyTorch default (1e-8) for stability
adam_beta1     = 0.9
adam_beta2     = 0.999
clip_grad_norm = 30         # Gradient clipping (value also clipped at 1000)
```

### Target network update
```python
soft_update_tau = 0.02      # θ_target ← 0.02·θ_online + 0.98·θ_target
# Updated every 2048 training samples
```
Soft updates are more stable than hard updates (periodic full copy). τ=0.02 means the target network tracks the online network slowly, preventing target Q oscillations.

### IQN-specific settings
```python
iqn_embedding_dimension = 64   # Cosine basis dimensionality for τ embedding
iqn_n = 8                      # Quantile samples per training step (must be even)
iqn_k = 32                     # Quantile samples at inference (average for action selection)
iqn_kappa = 5e-3               # Huber threshold in quantile loss
use_ddqn = False               # Double DQN: disabled (IQN's distributional loss already reduces overestimation)
```

### Game timing settings
```python
ms_per_action               = 50    # One action = 50 ms game time
tm_engine_step_per_action   = 5     # 5 engine steps × 10 ms = 50 ms
temporal_mini_race_duration_ms = 7000   # Episode length = 7 seconds
temporal_mini_race_duration_actions = 140  # 7000 / 50 = 140 actions per episode
```
Episodes are 7-second "mini-races": the car starts, drives for 7 seconds, then resets regardless of whether it finished the track. This keeps data collection efficient (no waiting for the agent to finish or crash on a 30-second track).

---

## Data Flow — One Training Step End to End

```
1. TMNF game advances 50 ms
2. TMInterface sends (BGRA frame, game state floats) to collector
3. Collector:
   a. Converts frame: BGRA → greyscale → (1, 120, 160) uint8
   b. Reads 127 float features from game state
   c. Calls IQN.infer_network(frame, floats, iqn_k=32)
      → gets Q(s, a, τ) for 32 quantiles, all 12 actions
      → averages over τ → mean Q(s, a) for each action
      → returns argmax_a Q(s, a) as the selected action
   d. Sends action (keyboard keypress bits) back to TMInterface
   e. Stores (frame, floats, action, reward) in experience replay buffer

4. Learner (running in parallel):
   a. Samples 512 transitions from replay buffer
   b. For each transition (s, a, r, s'):
      - Compute n-step return G = r_t + γr_{t+1} + ... + γ^n·Q_target(s_{t+n}, a*)
      - Sample iqn_n=8 quantiles τ
      - Predict Q_online(s, a, τ) using online network
      - Compute pinball Huber loss
   c. Adam optimizer step
   d. Every 2048 samples: soft-update target network
   e. Log avg_Q, single_zones_reached, loss to TensorBoard
```

---

## TensorBoard Metrics — What to Watch

| Metric | What it means | Healthy behavior |
|--------|--------------|-----------------|
| `avg_Q` | Mean Q-value across the buffer | Rises steadily; plateauing = agent converged |
| `single_zones_reached` | How many VCP zones the agent reaches per episode (higher = further along the track) | Increases episode by episode; should hit full track value once laps are completed |
| `loss` | IQN quantile regression loss | Should fall and stabilize; spiking = instability |
| `epsilon` | Current random exploration rate | Should follow the schedule: 1.0 → 0.03 |
| `mean_race_time` | Mean time for completed episodes | Should decrease as agent gets faster |

**Stop training when:**
- `single_zones_reached` has been at its maximum value (full track) for 500k+ steps
- `mean_race_time` is no longer improving
- You have at least 20 clean lap replays to use for evaluation

---

## Key Differences from Phase 1 (TMRL/SAC)

| Dimension | Phase 1 — TMRL/SAC | Phase 2 — Linesight/IQN |
|-----------|---------------------|--------------------------|
| Game | Trackmania 2020 | Trackmania Nations Forever |
| Observation | 19 LIDAR rays × 4 frames | 160×120 image + 127 float vector |
| Algorithm | SAC (continuous actions, actor-critic) | IQN (discrete actions, distributional Q-learning) |
| Action space | 3 continuous floats (gas, brake, steer) | 12 discrete keyboard combinations |
| Reward | Per-step centerline progress | Time penalty + distance + potential shaping |
| Episode length | Full lap (no cutoff) | 7-second mini-races |
| Convergence issue | Wall-hugging local minimum | N/A (converges cleanly on apex-bone) |
| Training convergence | 13 iterations across ~3 days | Fast convergence on simple track |

---

## File Map (Linesight Source)

```
linesight/
├── config_files/
│   ├── config.py               ← ALL hyperparameters (schedules, network dims, reward weights)
│   ├── config_copy.py          ← Runtime copy, reloaded live during training
│   ├── inputs_list.py          ← The 12 discrete actions (keyboard combinations)
│   ├── state_normalization.py  ← Per-feature mean/std for float vector normalization
│   └── user_config.py          ← Local paths (game install, username, port numbers)
├── trackmania_rl/
│   ├── agents/
│   │   └── iqn.py              ← IQN_Network class, loss function, infer_network()
│   ├── multiprocess/
│   │   ├── collector_process.py  ← Game rollout: runs episodes, fills buffer
│   │   ├── learner_process.py    ← Training loop: samples buffer, gradient steps
│   │   └── debug_utils.py
│   ├── tmi_interaction/
│   │   ├── game_instance_manager.py  ← Screen capture, float state reading, action sending
│   │   └── tminterface2.py           ← Low-level TMInterface socket protocol
│   ├── buffer_management.py    ← Reward computation, n-step return construction
│   ├── buffer_utilities.py     ← Replay buffer construction and sampling
│   ├── reward_shaping.py       ← Speedslide quality function (for engineered bonuses)
│   └── experience_replay/
│       └── experience_replay_interface.py  ← Experience namedtuple definition
└── scripts/
    └── train.py                ← Entry point: launches collector + learner processes
```
