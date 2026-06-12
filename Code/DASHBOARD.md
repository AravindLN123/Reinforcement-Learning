# Project Apex Control Dashboard — User Manual

The dashboard is a single-operator Streamlit app that wraps Linesight's training loop. It lets you start/stop training, tweak hyperparameters live, take snapshots of the model, watch the agent race, and detect a plateau automatically — all in one browser tab.

It's deliberately one file (`dashboard.py`) plus a one-line extras list (`requirements-dashboard.txt`). If you ever decide you don't want it, two `git rm` commands remove it cleanly with no impact on Linesight.

---

## Setup (one-time)

In the project venv:

```powershell
cd Code
.venv\Scripts\activate
pip install -r requirements-dashboard.txt
```

This installs `streamlit` (plus its transitive deps). It does not touch the training stack.

---

## Launch

```powershell
streamlit run dashboard.py
```

Streamlit opens http://localhost:8501 in your default browser. Leave the terminal running — closing it shuts down the dashboard server.

---

## The main page

![Main page](../figures/dashboard/MainPage.png)

Top to bottom:

- **Status bar** — current training state, run name, frames-played counter, and four action buttons.
- **Tabs** — Hyperparameters, Metrics, Snapshots, Race the agent, Logs.
- **Auto-refresh** — while training is running, the page reloads every 5 seconds so charts and the frames counter stay live.

### Status bar (zoomed)

![Status bar](../figures/dashboard/status_bar.png)

| Element | Meaning |
|---|---|
| **🟢 Running / ⚪ Stopped** badge | Is a training subprocess alive? |
| **Run name** | The active run, read from the live `config_copy.run_name`. All saves and TB logs live under `linesight/save/<run_name>/` and `linesight/tensorboard/<run_name>*/`. |
| **Frames played** | Total environment steps so far, read from `accumulated_stats.joblib`. |
| **▶ Start** | Launches `linesight/scripts/train.py` as a subprocess. Greyed out if a run is already alive. |
| **■ Stop** | Hard-kills the training process tree (`taskkill /F /T /PID …`) and any `TmForever.exe` instances. Mirrors what Linesight's own SIGINT handler does. Per-iteration checkpoints mean Stop loses at most one iteration. |
| **📸 Snapshot** | Freezes the current weights into a timestamped subfolder (see Snapshots tab below). |
| **↻ Refresh** | Clears the 5-second TB cache and re-reads everything. Useful if you edited config files externally. |

---

## Hyperparameters tab

![Hyperparameters tab — schedules and rewards](../figures/dashboard/ZoomedMainPage.png)

![Map cycle + Advanced internals](../figures/dashboard/MapCycleAndAdvancedDeepInternals.png)

**While training is running**, the banner at the top reads:
> Training is running. Changes write to **config_copy.py** for live hot-reload (picked up within ~1 iteration) **and** config.py so they survive restart.

That's because Linesight's learner and collector loops call `importlib.reload(config_copy)` every iteration ([learner_process.py:238](../linesight/trackmania_rl/multiprocess/learner_process.py#L238), [collector_process.py:81](../linesight/trackmania_rl/multiprocess/collector_process.py#L81)). The dashboard atomically writes both files so changes take effect within a second or two without losing the replay buffer.

**While training is stopped**, only `config.py` is updated; `config_copy.py` is regenerated automatically on next Start.

### Sections in this tab

| Section | What it does |
|---|---|
| **↺ Reset to defaults** | First dashboard load freezes the current `config.py` as `config.default.py`. The button restores it (also writes to `config_copy.py` if training is running). Frozen-date caption next to the button. |
| **Schedules** | Learning rate, epsilon, epsilon-Boltzmann, gamma. Each shows the current scheduled value at the current step, plus a "Flatten from now" button that rewrites the schedule to a constant from now onward. |
| **Reward weights** | `constant_reward_per_ms` (time penalty) and `reward_per_m_advanced_along_centerline`. Apply button writes immediately. |
| **Engineered rewards** | Speedslide / neoslide / kamikaze / close-to-VCP — flat values that get rewritten as `[(0, value)]` schedules. |
| **Game speed** | `running_speed` — percentage of real-time the game runs at (e.g. `20` = slow-motion for recording, `100` = real-time, `400` = 4x speed). Hot-reloaded every lap via `config_copy.py`; no restart needed. Does not affect training quality, only wall-clock throughput. |
| **Map cycle** | Multi-select of maps the agent trains on. 4 explo + 1 eval per map. New track? See [ADD_NEW_TRACK.md](ADD_NEW_TRACK.md). |
| **Advanced** (collapsed) | batch_size, IQN params (`iqn_n`, `iqn_k`, `iqn_kappa`), memory size schedule. IQN params are marked "restart recommended" because changing quantile counts mid-run isn't validated. |

---

## Metrics tab

![Metrics tab](../figures/dashboard/MatricsTab.png)

Live TensorBoard scalars, no separate TensorBoard server required. The dashboard walks every suffix-rotated TB dir for the active run (`figure8_run01`, `figure8_run01_2`, `figure8_run01_3`, …) and stitches the events back into a single time-series per metric.

Six panels by default: `loss`, `eval_race_time_robust`, `eval_race_time_finished`, `explo_race_time_finished`, `avg_Q`, `single_zone_reached`. For each, the dashboard tries the bare tag first, then map-suffixed variants (`_trained_fig8`, `_trained_monza`, `_trained_oval1`).

Charts are cached for 5 seconds — that's the autorefresh cadence too. Click ↻ Refresh in the status bar to force-bust the cache.

---

## Snapshots tab

![Snapshots tab](../figures/dashboard/SnapshotsTab.png)

### Manual snapshot

Type an optional note (e.g. `best-lap-43s` or `before-lr-tweak`) and click **📸 Take snapshot now**. The dashboard copies the five checkpoint files —

```
weights1.torch, weights2.torch, optimizer1.torch, scaler.torch, accumulated_stats.joblib
```

— from `linesight/save/<run_name>/` into a new folder under `linesight/save/<run_name>/snapshots/<timestamp>_step<N>[_<note>]/`.

The point: Linesight's checkpoint files are overwritten every iteration. If you see a great result, snapshotting freezes that moment so future training can't degrade it.

### Plateau auto-stop

Tick the checkbox to enable. While training is running and the dashboard tab is open in your browser, the 5-second autorefresh loop reads `eval_race_time_robust` events across all TB dirs, finds the best (min) value, and compares its `wall_time` against the latest event. If no improvement has been recorded for `PLATEAU_PATIENCE_MINUTES` minutes (default 30) of training wall-clock, it:

1. Takes a snapshot tagged `auto_plateau`
2. Calls Stop (clean kill + TM cleanup)
3. Shows a persistent yellow banner in the status bar until you click **Dismiss**

The caption under the checkbox shows the current state ("Best X. Y minutes since last improvement").

**Important:** the plateau check runs inside Streamlit's rerun loop. If you close the browser tab, the dashboard server stays up but the rerun stops — auto-stop won't fire. Keep the tab open if you want hands-off plateau detection.

### Existing snapshots list

All snapshots for the current run are listed below, newest first, with timestamps and total size in MB. The folder path is shown so you can navigate to it in Explorer.

---

## Race the agent tab

![Race the agent tab](../figures/dashboard/RaceTheAgentTab.png)

This tab replays a recorded "best lap" inside TMNF so you can:

1. **Watch the agent drive** — see what the best lap actually looks like in the game.
2. **Race against the agent as a ghost** — load the saved `.Replay.Gbx` in normal TMNF and try to beat it.

### Playback procedure (the order matters)

1. **Stop training** — TMNF can only host one process at a time.
2. **Click ▶ Press Playback** on the run you want to replay. _Don't launch TMNF yet._ The dashboard now shows ⏳ "Playback loading — now launch the game" with a live spinner.
3. **Now** launch TMNF via TMLoader (the same way you start training). Pick your profile when prompted.
4. The dashboard auto-detects the TMInterface connection and the status flips through 🔌 Connected → 🗺️ Map loading → 🏎️ Agent is driving.
5. When the lap finishes, TMNF auto-saves `.Replay.Gbx` to `~/Documents/TrackMania/Tracks/Replays/`, and the dashboard also copies it next to the inputs file at `linesight/save/<run>/best_runs/<run>_replays/`.

**Why this order:** the script enters a 20-second connection retry loop before issuing any TMInterface commands. Clicking Playback first means the connection succeeds cleanly after TMNF reaches a stable menu state. Reverse order can cause the `map` command to hit mid-profile-selection and stall.

### Race against the agent as a ghost

1. Do the playback first (so a `.Replay.Gbx` exists in TMNF's replay folder).
2. Close TMLoader-TMNF.
3. Launch **regular TMNF from Steam** (not TMLoader — TMLoader is for training only).
4. In TMNF: **Solo → Race against replay → pick the `.Replay.Gbx` → start.**
5. The agent appears as a translucent ghost car you can try to beat.

### Best runs list

Sorted by lap time, fastest first. Filter by map dropdown at the top. Each row shows the map short name, lap time formatted as `M:SS.mmm`, folder path, inputs file, and a ▶ Playback button.

The button is greyed out if training is currently running (would conflict) or if another playback is already in progress.

### Playback log

Live tail of `dashboard_playback.log` at the bottom of the tab. Watch this if a playback isn't behaving — it shows every TMInterface message and any errors from the subprocess.

---

## Logs tab

![Logs tab](../figures/dashboard/LogsTab.png)

Tail of `dashboard_training.log` — the captured stdout + stderr of the training subprocess. Last ~20 KB shown, refreshed on every page render. Useful for confirming Linesight loaded the right checkpoint on Start, watching the agent's startup banner, or diagnosing crashes.

---

## Common workflows

### Resume after a crash or Ctrl+C

1. Make sure TMNF and any leftover Python processes are gone (Task Manager → end task on `python.exe`, `TmForever.exe` if they're still around).
2. Open the dashboard.
3. Click ▶ Start. Linesight reads `weights1.torch`, `optimizer1.torch`, `accumulated_stats.joblib` from the run's save folder and resumes from the exact step the last checkpoint was written. No special flag — auto-resume is default behavior.

### Try a different learning rate without losing the buffer

1. Hyperparameters tab → Schedules → Learning rate → type the new value → **Flatten from now**.
2. The schedule is rewritten to keep the current value until now, then jump to the new value. Both `config.py` and `config_copy.py` are updated atomically. Within ~1 iteration the learner's optimizer LR updates ([learner_process.py:267](../linesight/trackmania_rl/multiprocess/learner_process.py#L267)).

### Swap to a different map mid-training

1. Hyperparameters tab → Map cycle → multi-select → check the new map (uncheck old ones).
2. Click **Apply map cycle**.
3. The collector detects the change at [collector_process.py:88-91](../linesight/trackmania_rl/multiprocess/collector_process.py#L88-L91) and switches maps on its next loop iteration. Replay buffer is preserved — useful for transfer-learning between similar tracks.

### Freeze a great moment

1. Status bar → **📸 Snapshot** (one-click, timestamp only).
2. Or Snapshots tab → type a note → **Take snapshot now**.
3. The frozen checkpoint is now safe; future training can keep going.

### Roll back a hyperparameter experiment that broke training

1. Hyperparameters tab → top right → **↺ Reset to defaults**.
2. Confirms restoration from `linesight/config_files/config.default.py` (frozen on your first dashboard launch).

### Record a slow-motion video of the agent

1. Hyperparameters tab → **Game speed** → set `running_speed` to `20` → **Apply**.
   The game drops to 20% of real-time (slow-motion). The change takes effect within one lap — no restart needed.
2. Use your screen recorder (OBS, Windows Game Bar, etc.) to capture the next playback or training run.
3. When done, set `running_speed` back to `100` (real-time) or your preferred training speed.

> The `running_speed` setting is purely cosmetic for the game window — it does not change the number of environment steps per second the agent experiences. Training quality is unaffected.

### Add a brand new track

See the full procedure in [ADD_NEW_TRACK.md](ADD_NEW_TRACK.md). Summary: install `.Challenge.Gbx`, drive one clean reference lap, run `gbx_to_vcp.py`, add the map to `KNOWN_MAPS` (dashboard.py) and `MAP_PATH_BY_SHORT` (playback_best_run.py), then use the dashboard normally.

---

## Files the dashboard creates

| Path | Purpose | Tracked in git? |
|---|---|---|
| `.dashboard_training.pid` | PID of the running training subprocess | No (in `.gitignore`) |
| `.dashboard_playback.pid` | PID of the running playback subprocess | No (in `.gitignore`) |
| `dashboard_training.log` | Captured stdout/stderr of training | No (`*.log`) |
| `dashboard_playback.log` | Captured stdout/stderr of playback | No (`*.log`) |
| `linesight/config_files/config.default.py` | Frozen defaults — created on first launch | Inside `linesight/` which is gitignored from the outer repo |
| `linesight/save/<run>/snapshots/<timestamp>_step<N>[_<note>]/` | Frozen checkpoint snapshots | Inside `linesight/save/` — managed by Linesight's own gitignore |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Status reads ⚪ Stopped but training appears to be running | The training process was started outside the dashboard (terminal). The dashboard only tracks runs it started via ▶ Start. | Either stop the manual run and start via the dashboard, or just use the dashboard's read-only panes (Metrics, Logs, Snapshots) — they still work regardless of who started training. **Never click ▶ Start while a manual training run is alive** (would launch a second instance and crash both). |
| Playback shows ⏳ loading forever | TMNF wasn't launched, or TMLoader isn't attaching TMInterface | Check that TMLoader is configured to load TMInterface 2.1.0 (see [INSTALL.md](../INSTALL.md)). Verify port 8478 isn't held by another process. |
| Playback connects then times out at "🗺️ Map loading" | TMNF has a personal-best replay autosave that's blocking the race-start handshake | The playback script handles this automatically by renaming the autosave to `.bak`. If it's failing, check `dashboard_playback.log` for the `[hide_pb]` line. The TMNF profile username must match `user_config.username`. |
| Plateau never fires even after long flat periods | Browser tab was closed | Streamlit only runs while a client is connected. Keep the tab open for auto-stop. |
| Reset to defaults doesn't work | `config.default.py` was deleted | Delete `linesight/config_files/config.default.py` if it exists, then reload the dashboard — it will freeze the current `config.py` as the new defaults. |
| Subprocess prints "UnicodeEncodeError: cp1252" | Non-ASCII chars in a `print()` statement of a child script | The dashboard sets `PYTHONIOENCODING=utf-8` for all subprocesses it spawns, so this shouldn't happen for dashboard-launched processes. If it does, check that the subprocess is going through `_subprocess_env()` in dashboard.py. |

---

## Where to read more

- [docs/ADD_NEW_TRACK.md](ADD_NEW_TRACK.md) — full procedure for teaching a brand new track.
- [QUICKSTART.md](QUICKSTART.md) — step-by-step project workflow (the dashboard is used from Step 5 onward).
- [LINESIGHT_EXPLAINED.md](LINESIGHT_EXPLAINED.md) — what's inside the model the dashboard controls.
- `dashboard.py` itself — single file, heavy comments. Worth a read if you're extending it.
