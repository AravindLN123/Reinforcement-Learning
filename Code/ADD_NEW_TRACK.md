# Adding a brand new track for the agent to learn

This is the full end-to-end procedure for teaching the Project Apex agent a new Trackmania track. The dashboard handles all the *training control*, but adding a brand new track requires a handful of one-time steps outside the dashboard first. After those, the dashboard takes over.

Estimated time: **15-30 minutes** of setup, then training time depends on the track (a simple oval converges in hours; a complex track can take many hours or days).

---

## What you need before starting

- The dashboard already working on your machine (see [QUICKSTART.md](../QUICKSTART.md)).
- TMLoader + TMInterface installed and the `labHSE` (or your) TMNF profile configured.
- A working copy of TMNF that you can drive in solo mode.
- A `.Challenge.Gbx` file for your new track (either build one in TMNF's editor or download from [tm-exchange.com](https://tm-exchange.com)).

---

## Step 1 — Install the map file in TMNF

Copy your `.Challenge.Gbx` to:

```
C:\Users\<you>\Documents\TrackMania\Tracks\Challenges\My Challenges\<MapName>.Challenge.Gbx
```

Launch TMNF, go to **Solo → My Challenges**, and verify the map loads and you can drive it. If TMNF refuses to load the map, the rest of this procedure won't work — fix the map file first.

---

## Step 2 — Drive one clean reference lap

Linesight learns by following a **reference trajectory** (a recorded ideal racing line). It doesn't need to be fast — it just needs to be a smooth, complete lap on the racing line you want the agent to follow.

1. In **regular TMNF launched from Steam** (not TMLoader — TMLoader is for training only), open your new map in solo.
2. Drive one full lap. Try to stay on the racing line. Speed doesn't matter; clean trajectory does.
3. When you finish, TMNF auto-saves a replay to:
   ```
   C:\Users\<you>\Documents\TrackMania\Tracks\Replays\Autosaves\<TMNFusername>_<MapInternalName>.Replay.gbx
   ```
   The "internal name" is usually the same as the file name (`Figure8Track`, `ovaltrack1`), but for some downloaded maps it can differ (Monza's file is `Monza.Challenge.Gbx` but its internal name is `GT12_Monza_Circuit`).

If the lap was rough, drive again — TMNF overwrites the autosave with each clean run.

---

## Step 3 — Generate the reference centerline `.npy`

Linesight has a tool that converts your replay into a sequence of "virtual checkpoints" sampled every 50 cm along the racing line. This is what the agent learns to follow.

From the repo root, in your training venv:

```powershell
cd linesight
python scripts\tools\gbx_to_vcp.py "C:\Users\<you>\Documents\TrackMania\Tracks\Replays\Autosaves\<YourReplay>.Replay.gbx"
```

This writes a file `linesight/maps/map.npy`. **Rename it** to match the project's naming convention:

```powershell
mv linesight\maps\map.npy linesight\maps\<MapName>_0.5m_cl.npy
```

(The `0.5m_cl` suffix is shorthand for "centerline sampled every 50 cm" — keep it consistent with the existing maps so the tooling finds it.)

---

## Step 4 — Register the new map with the dashboard and playback

Two small dictionaries to extend — each is one new line:

**In [dashboard.py](../dashboard.py), find `KNOWN_MAPS`** (near the top, around line 75):
```python
KNOWN_MAPS: list[tuple[str, str, str]] = [
    ("monza",      '"My Challenges/Monza.Challenge.Gbx"',         "Monza_0.5m_cl.npy"),
    ("fig8",       '"My Challenges/Figure8Track.Challenge.Gbx"',  "Figure8Track_0.5m_cl.npy"),
    # ... existing entries ...
    ("newmap",     '"My Challenges/NewMap.Challenge.Gbx"',         "NewMap_0.5m_cl.npy"),  # ← add this
]
```

This makes the new map appear in the Map cycle multi-selector in the dashboard.

**In [playback_best_run.py](../playback_best_run.py), find `MAP_PATH_BY_SHORT`** (near the top):
```python
MAP_PATH_BY_SHORT = {
    "monza":      '"My Challenges/Monza.Challenge.Gbx"',
    # ... existing entries ...
    "newmap":     '"My Challenges/NewMap.Challenge.Gbx"',  # ← add this
}
```

This lets the dashboard's "Race the agent" tab find the right map when you click Playback on a `newmap_xxxxx` recorded run.

> **The `short name` is your choice** — use a short identifier (no spaces, no special chars). It will appear in best-run folder names like `newmap_42500/`.

---

## Step 5 — Choose fresh agent vs. transfer learning

Decide before starting training:

| Option | Description | When to pick it |
|---|---|---|
| **Fresh agent** | Change `run_name` in `linesight/config_files/config.py` to e.g. `newmap_run01`. New checkpoint dir under `save/`, new TB dir, network starts from random weights. | You want a clean scientific result — the agent's performance is fully attributable to the new map. Slower convergence. |
| **Transfer learning** | Keep `run_name` as-is (e.g. `figure8_run01`). The agent continues with its existing weights and replay buffer, just adds the new map to the rotation. | You want the agent to converge faster by leveraging skills learned on previous tracks. Conflates training history across maps. This is what `figure8_run01` already does — it was trained on Figure8Track first, then Monza was added. |

Project Apex's existing `figure8_run01` is a transfer-learning example. The choice depends on your goal — both are valid.

---

## Step 6 — Pick the new map in the dashboard

1. Open the dashboard at http://localhost:8501.
2. Go to the **Hyperparameters** tab → **Map cycle** section.
3. In the multi-select, check the new map. Optionally uncheck other maps you don't want in the rotation.
4. Click **Apply map cycle**.
   - If training is currently running, the change writes to `config_copy.py` and the collector picks up the new cycle on its next loop iteration (no buffer loss).
   - If training is stopped, the change writes to `config.py` and applies on the next **▶ Start**.

---

## Step 7 — Start training

Click **▶ Start** in the dashboard's top bar. Watch the **Metrics** tab:

- `single_zone_reached_trained_<short>` is the leading indicator early — it climbs from 0 → full track length as the agent learns to reach further along the track.
- Once the agent consistently finishes, `eval_race_time_robust_trained_<short>` (lap time) becomes the metric to watch.
- `loss` and `avg_Q` should be increasing/stable. Loss spikes are normal; sustained explosion is not.

For a simple track (oval, figure-8), expect first finishes in 1–3 hours and a usable best lap in 6–12 hours of training. For complex tracks (curves, jumps, obstacles), allow days.

---

## Step 8 — When the agent plateaus

Enable **auto-stop** in the Snapshots tab if you want the dashboard to detect a plateau and freeze a snapshot automatically. Otherwise, watch the metrics yourself and click **📸 Snapshot** when you see a result you want to preserve, then **■ Stop**.

To see the agent drive your new map: **Race the agent** tab → filter by your new short name → click **▶ Press Playback** → launch TMNF via TMLoader → pick your profile. (See the instructions inside that tab for the rest of the racing-against-ghost flow.)

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Training starts but the agent never moves | `gbx_to_vcp.py` didn't run on the right replay, or `.npy` is missing | Re-check Step 3. The `.npy` file must exist at `linesight/maps/<MapName>_0.5m_cl.npy` and the entry in `KNOWN_MAPS` must point to that exact filename. |
| Playback loads the wrong map | The run folder's short name isn't in `MAP_PATH_BY_SHORT` | Re-check Step 4. The short name (the part before the underscore in folder names like `newmap_42500/`) must match a key in `MAP_PATH_BY_SHORT`. |
| Playback hangs at "🗺️ Map loading…" | TMNF has a personal-best replay autosaved for the new map that's blocking the race-start handshake | The playback script handles this automatically (it renames `*_<MapName>.Replay.gbx` to `.bak`), but if the autosave is in an unexpected location, do it manually. See [playback_best_run.py](../playback_best_run.py) → `hide_pb_replay()`. |
| Best laps look terrible even after long training | The reference centerline is bad (rough or off the racing line) | Drive a better reference lap in Step 2, regenerate the `.npy` in Step 3. |
| `Unknown map short name 'newmap'` when clicking Playback | Forgot Step 4 | Add the entry to `MAP_PATH_BY_SHORT`. |

---

## Summary

| Step | Action | Tool |
|---|---|---|
| 1 | Install `.Challenge.Gbx` in TMNF | File copy |
| 2 | Drive one clean reference lap | TMNF (Steam, not TMLoader) |
| 3 | Generate `.npy` reference centerline | `python scripts\tools\gbx_to_vcp.py` |
| 4 | Add map entry to `KNOWN_MAPS` + `MAP_PATH_BY_SHORT` | Code edit (2 lines total) |
| 5 | Decide fresh agent vs. transfer learning | Edit `run_name` in config.py (or don't) |
| 6 | Pick the map in the dashboard | Dashboard → Hyperparameters → Map cycle |
| 7 | Click ▶ Start | Dashboard |
| 8 | Watch metrics, snapshot at peak | Dashboard |

Steps 1–4 are one-time per new track. After that, training/snapshotting/playback runs entirely through the dashboard.
