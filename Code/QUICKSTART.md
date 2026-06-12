# Quickstart

Once [INSTALL.md](INSTALL.md) is complete, this guide walks you from a fresh install to a recorded ghost replay.

---

## Step 1 — Install and verify

Follow [INSTALL.md](INSTALL.md). Confirm the verification script passes before continuing.

---

## Step 2 — Build your track in TMNF

The TMNF map editor is older than TM2020's and works differently. Plan for 30-45 minutes the first time.

1. Launch TMNF via TMLoader.
2. **Create** → **New Track** → pick an environment (stadium gives the cleanest visuals).
3. Design your layout. A few principles that help the agent learn:
   - Prefer gentle curves over sharp 90-degree turns (sharp turns make wall-hugging easier)
   - Avoid loops, jumps, and elevation changes (sparse reward signal)
   - Keep the track wide (more room for racing-line variation)
   - Simple closed loops work well; figure-8 shapes also work
4. Place the **start block** and a **lap finish line** at the same position (closed-loop track).
5. **Validate** the track in the editor (press the green-flag button) and drive a slow lap yourself to confirm the car can complete it cleanly.
6. **Save the track** as `TrackName` (TMNF stores it in `My Documents\TmForever\Tracks\` or similar).
7. **Copy the .Gbx file** into this project's `maps/` folder:
   ```powershell
   copy "$env:USERPROFILE\Documents\TmForever\Tracks\Challenges\Custom\TrackName.Challenge.Gbx" Code\maps\
   ```
   (TMNF calls map files `.Challenge.Gbx` rather than `.Map.Gbx`; both work with TMInterface.)

---

## Step 3 — Record the reference trajectory

Linesight learns by following a centerline drive you record. It does not need to be fast -- it needs to be clean (no wall scrapes, no off-track).

1. Launch TMNF via TMLoader. Load `TrackName`.
2. **Drive a slow centerline lap** (keyboard arrow keys or a controller):
   - Stay in the geometric middle of the track
   - 30-50% throttle; do not try to be fast
   - Avoid walls completely
   - Cross the finish line at the end of one lap
3. TMNF auto-records the replay. After finishing, save it via the "Save Replay" menu.
4. Find the replay file (typically `My Documents\TmForever\Tracks\Replays\`). Copy it into `maps/`:
   ```powershell
   copy "$env:USERPROFILE\Documents\TmForever\Tracks\Replays\<your_replay>.Replay.Gbx" Code\maps\TrackName_centerline.Replay.Gbx
   ```

---

## Step 4 — Convert the replay to a reference trajectory

Linesight ships a converter script. From your activated Python venv, run it from the `linesight/` folder:

```powershell
cd Code\linesight
python scripts\tools\gbx_to_vcp.py ..\maps\TrackName_centerline.Replay.Gbx
```

This produces `maps/map.npy`. Rename it following Linesight's naming convention:

```powershell
move ..\maps\map.npy ..\maps\TrackName_0.5m_cl.npy
```

(`cl` = centerline -- Linesight uses this suffix to tag the trajectory type.)

### Optional -- visualize the reference line in-game

1. Launch TMNF with TMInterface on port 8477:
   ```powershell
   # In TMLoader, add launch argument: /configstring="set custom_port 8477"
   ```
2. Load `TrackName`, then run:
   ```powershell
   python scripts\tools\tmi2\add_vcp_as_triggers.py ..\maps\TrackName_0.5m_cl.npy -p 8477
   ```
3. The trajectory should overlay the track surface where you drove. If it floats above the track or veers off, re-record Step 3.

### Register the track in Linesight's config

Edit `linesight/config_files/config.py`:

1. Find the `map_cycle` variable.
2. Add an entry for your track pointing at your map and reference files. Follow the same format as existing entries:
   ```python
   map_cycle = [
       {
           "name": "TrackName",
           "map_path": "../maps/TrackName.Challenge.Gbx",
           "reference_line": "../maps/TrackName_0.5m_cl.npy",
       },
       # ...other maps...
   ]
   ```
3. Save the file.

---

## Step 5 — Start training

> Before starting, make sure you have several hours of uninterrupted training time, the PC is on a stable power supply, and Windows sleep and screen-saver are disabled (Settings → System → Power → Sleep → Never).

1. Launch TMNF via TMLoader, load `TrackName`, leave the car at the start.
2. In a separate terminal, activate the venv and start training from the `linesight/` folder:
   ```powershell
   cd Code\linesight
   python scripts\train.py
   ```
3. In another terminal, start TensorBoard:
   ```powershell
   tensorboard --logdir runs
   ```
   Open `http://localhost:6006` in your browser. You can also monitor metrics from the dashboard (see below).

**What to watch during training:**

| Time | Expected signal |
|---|---|
| 30 min | `single_zones_reached` > 0 (agent reaching the first checkpoints) |
| 2 h | `single_zones_reached` growing (agent reaching further each episode) |
| 6 h | Agent regularly reaching the full track length |
| 12-24 h | Refining lap time, finishing consistently |

Training time varies by track complexity and GPU. Simple oval tracks converge in a few hours; complex circuits may take 24 hours or more.

### Use the dashboard while training

The **Project Apex Control Dashboard** lets you monitor and control training from a browser tab without touching the terminal. Start it with:

```powershell
cd Code
streamlit run dashboard.py
```

Open `http://localhost:8501`. From the dashboard you can start/stop training, tune hyperparameters live, snapshot weights, and watch the agent race. Full guide: [DASHBOARD.md](DASHBOARD.md).

---

## Step 6 — Save the best lap replay

After training stops (or you press Ctrl+C when satisfied):

1. The best checkpoint and `.inputs` file are saved automatically under `linesight/save/<run_name>/best_runs/`.
2. To convert the best `.inputs` file to a `.Replay.Gbx` that TMNF can load as a ghost:
   - Launch TMNF via TMLoader and load the track.
   - In the TMInterface console (press F3 in-game), run:
     ```
     set scripts_folder Code\linesight\save\<run_name>\best_runs\<run_folder>
     load <filename>.inputs
     ```
   - Wait for the run to complete -- TMNF saves it as a personal best replay automatically.
3. Copy the replay from `Documents\TrackMania\Tracks\Replays\` into `results/`:
   ```powershell
   copy "$env:USERPROFILE\Documents\TrackMania\Tracks\Replays\<map>_labHSE(*).Replay.Gbx" Code\results\
   ```

---

## Step 7 — Race against the agent ghost

The AI ghost is the personal best replay TMNF saved when you played back the best `.inputs` file. Racing against it shows a human vs. AI comparison in real time.

> **Important:** TMNF's Solo → campaign screen is for the official campaign only and does not list custom tracks. Use the Track Editor route instead.

1. Launch TMNF via TMLoader.
2. Main menu → **Track Editor**.
3. **Edit a Track** → select `TrackName` from the list.
4. Once the track loads, click the **Test** button (the green play/flag icon at the top).
5. The AI ghost appears automatically as a semi-transparent car (TMNF loads it as the saved personal best for this map).
6. Press **Delete** to restart with a proper 3-2-1 countdown -- you and the ghost both start from the same point. Use Delete any time to restart cleanly.

### If the ghost does not appear

1. TMNF main menu → **Replays** → **My Replays**.
2. Find your replay for `TrackName`.
3. Right-click → **Set as Best**.
4. Re-enter Track Editor → Test -- the ghost should now appear.

---

## Quick reference

| What you want | Command / Action |
|---|---|
| Activate Python venv (conda) | `conda activate linesight` |
| Activate Python venv (venv) | `.venv\Scripts\activate` |
| Install the dashboard (one-time) | `pip install -r requirements-dashboard.txt` |
| Launch the dashboard | `streamlit run dashboard.py` → http://localhost:8501 |
| Convert replay to reference | `python scripts\tools\gbx_to_vcp.py <path>.Replay.Gbx` |
| Visualize reference in-game | `python scripts\tools\tmi2\add_vcp_as_triggers.py <path>.npy -p 8477` |
| Start training (terminal) | `python scripts\train.py` (from `linesight/`) |
| Start training (dashboard) | Click Start in the status bar |
| Monitor TensorBoard | `tensorboard --logdir runs` → localhost:6006 |
| Stop training cleanly (terminal) | Ctrl+C in the training terminal |
| Stop training cleanly (dashboard) | Click Stop in the status bar |
| Snapshot weights | Dashboard → Snapshots tab |
| Replay a recorded best lap | Dashboard → Race the agent tab → Playback |
| Add a new track | See [ADD_NEW_TRACK.md](ADD_NEW_TRACK.md) |
| Load track for demo | Track Editor → Edit a Track → TrackName → Test button |
| Restart race with countdown | Press Delete in-game |
| Ghost missing | Replays → My Replays → right-click replay → Set as Best |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `gbx_to_vcp.py` fails to parse the replay | Replay is from a different game version or was saved incorrectly | Re-record Step 3; complete a full lap and save via TMNF's save-replay menu |
| `add_vcp_as_triggers.py` shows trajectory off-track | Centerline drive was sloppy | Re-record Step 3 |
| Training does not connect to TMNF | TMInterface port mismatch | Confirm `custom_port` in `user_config.py` matches the port TMNF launches on |
| `single_zones_reached` stuck at 0 | Reference file path wrong in `config.py` | Double-check the path; paths are relative to `linesight/` |
| `avg_Q` is NaN | Numerical instability | Restart training; if persistent, check Linesight's troubleshooting docs |
| Training is slow (< 10 fps) | TMNF graphics quality too high | Reduce resolution in INSTALL.md Step 8 |
| Windows goes to sleep mid-training | Power settings | Settings → System → Power → Sleep → Never |
| Custom track not found in Solo menu | Solo menu only shows the official campaign | Use Track Editor → Edit a Track → select map → Test button |
| Ghost does not appear in Test mode | Personal best not set for this map | Replays → My Replays → right-click the best replay → Set as Best; then re-enter Test |
| Game speed too fast or slow after playback | TMInterface `set speed` command leftover | Open TMInterface console (F3) and type `set speed 1` |
| Race restarts with no countdown | Normal Test mode behavior | Press Delete to trigger a proper 3-2-1 countdown restart |

If you hit something not listed here, Linesight's docs have a Troubleshooting section and an active Discord community linked from their GitHub README.
