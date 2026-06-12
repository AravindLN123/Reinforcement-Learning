# Install Guide — Phase 2 (Linesight)

Setting up Trackmania Nations Forever, TMInterface, and Linesight from scratch. Allow ~2 hours including downloads.

> **Important — TMNF is a DIFFERENT game from TM2020.** TMNF is the older, free standalone Trackmania title that Linesight targets. You can have both installed simultaneously without conflicts. The Phase 1 TMRL project does not need to be uninstalled.

---

## Prerequisites checklist

- [ ] Windows 11 (Linux works too but this guide is Windows)
- [ ] ≥ 20 GB free disk space
- [ ] ≥ 20 GB RAM (Linesight requirement)
- [ ] NVIDIA GPU with CUDA 12.x support (RTX 5090 confirmed working in Phase 1)
- [ ] Admin access (some installers need it)

---

## Step 1 — Install Trackmania Nations Forever

TMNF is the older, free standalone Trackmania. The easiest source is **Steam** — search for "Trackmania Nations Forever" and install it from there (it's free).

After installing, launch the game from Steam. It will likely prompt you to download an update — let it complete. The game should start normally after that.

Note your Trackmania nickname — Linesight's user_config.py needs it later.

---

## Step 2 — Install ModLoader (TMLoader)

TMInterface needs TMLoader to inject into TMNF on launch.

1. Download from: https://donadigo.com/tminterface/  (look for "TMLoader" or "Mod Loader")
2. Run the installer. It will detect your TMNF install automatically.
3. After installation you'll have a `TmForeverLauncher.exe` (or similar) — this is what you launch from now on instead of the raw TMNF executable.

---

## Step 3 — Install TMInterface 2.1.0

1. Download TMInterface 2.1.0 from the same source: https://donadigo.com/tminterface/
2. Run the installer. It hooks into the TMLoader-managed TMNF install.
3. Verify by launching TMNF via TMLoader — you should see a small "TMInterface" message in the TM console at startup.

---

## Step 4 — Create a Python 3.11 virtual environment

> Linesight requires Python `>=3.10 and <3.12`. **The Phase 1 TMRL project uses 3.12, so this MUST be a separate venv.** Do NOT install Linesight into the same environment as TMRL.

Two options — pick one:

### Option A: Conda (recommended)

```powershell
conda create -n linesight python=3.11
conda activate linesight
```

### Option B: Standalone Python 3.11 + venv

1. Install Python 3.11 from python.org (separate from your existing 3.12)
2. Create venv inside the project folder:

```powershell
cd C:\Users\Wings\Documents\KarrarPrivate\ProjectApex-Linesight
py -3.11 -m venv .venv
.venv\Scripts\activate
```

Confirm: `python --version` → should print `Python 3.11.x`

---

## Step 5 — Install PyTorch with CUDA 12.x

> **RTX 50-series (Blackwell, sm_120) requires the cu128 wheels.** The cu121 wheels only ship kernels up to sm_90, so `torch.cuda.is_available()` will return True but every GPU op crashes with "no kernel image available." Use the cu128 index URL below for any RTX 50-series card. For older cards (RTX 30/40 series), cu121 is fine.

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify GPU — run a real tensor op, not just `is_available()`:

```powershell
python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available(), torch.version.cuda); print('Device:', torch.cuda.get_device_name(0)); x = torch.randn(1024, 1024, device='cuda'); print('GPU matmul OK, sum:', float((x @ x).sum().cpu()))"
```

Expected (RTX 5090): no warnings, prints device name, and `GPU matmul OK, sum: <number>`. If you see a warning about `sm_120 is not compatible`, you're on the wrong wheel — uninstall and reinstall from the cu128 index above.

---

## Step 6 — Clone Linesight into the project folder

From inside `ProjectApex-Linesight/`:

```powershell
git clone https://github.com/Linesight-RL/linesight.git
```

The `.gitignore` in this repo already excludes `linesight/`, so the clone won't be tracked by your repo.

---

## Step 7 — Install Linesight's Python dependencies

```powershell
cd linesight
pip install -e .
```

If `requirements_conda.txt` or similar is referenced in their README, follow that too. The exact dependencies file may have changed since this guide was written — check `linesight/README.md` for the current install procedure if `pip install -e .` errors.

---

## Step 8 — Configure TMNF for training

Launch TMNF via TMLoader. In-game:

1. Settings → Display → switch to **windowed mode**
2. Choose **the lowest resolution available** (e.g., 640×480 or whatever the minimum is)
3. Set graphics quality to **lowest** across all options
4. Disable shadows, vsync, motion blur, post-processing

Why: training is bottlenecked by game frame rate. Linesight wants as many frames per second as possible. Visual quality is irrelevant to the agent.

---

## Step 9 — Edit Linesight's user config

Open `linesight/config_files/user_config.py` in a text editor. The fields actually present are:

- `username` — your Trackmania nickname (case-sensitive; must match in-game name exactly)
- `target_python_link_path` — where `Python_Link.as` will be written into the TMInterface plugins folder. Default (`~/Documents/TMInterface/Plugins/Python_Link.as`) works for standard TMInterface installs.
- `trackmania_base_path` — TrackMania **user data** folder, *not* the install folder. Default (`~/Documents/TrackMania`) is correct on Windows.
- `base_tmi_port` — leave at `8478` unless that port is occupied.
- `windows_TMLoader_path` — path to `TMLoader.exe`. Default assumes `AppData\Local\TMLoader\TMLoader.exe`, but **if TMNF was installed via Steam, TMLoader lives in the Steam game folder** (e.g. `C:\Program Files (x86)\Steam\steamapps\common\TrackMania Nations Forever\TMLoader.exe`). Use `Path(r"...")` with a raw string to avoid backslash escaping issues.
- `windows_TMLoader_profile_name` — usually `"default"`. Confirm in TMLoader's `config.yaml` under `default_profiles:` if unsure.
- `linux_launch_game_path` — Windows users ignore.

Verify the file loads cleanly:

```powershell
cd C:\Users\Wings\Documents\KarrarPrivate\ProjectApex-Linesight\linesight
python -c "from config_files import user_config as u; print('TMLoader exists:', u.windows_TMLoader_path.exists()); print('TM base path exists:', u.trackmania_base_path.exists())"
```

Both should print `True`. Save the file.

---

## Step 10 — Verify everything works

End-to-end smoke test using [verify_tminterface.py](verify_tminterface.py), a one-shot script in this repo that connects, registers with TMInterface, reads a few telemetry frames, and exits with PASS/FAIL.

**Prerequisites (one-time setup before the test runs):**

1. Copy Linesight's `Python_Link.as` plugin into TMInterface's plugins folder:
   ```powershell
   Copy-Item "linesight\trackmania_rl\tmi_interaction\Python_Link.as" "$env:USERPROFILE\Documents\TMInterface\Plugins\Python_Link.as"
   ```
2. Tell TMInterface which port the plugin should listen on. Append to TMInterface's config:
   ```powershell
   Set-Content -Path "$env:USERPROFILE\Documents\TMInterface\config.txt" -Value "set custom_port 8478"
   ```
3. Generate the runtime config copy that Linesight's modules import (normally created by `train.py`):
   ```powershell
   Copy-Item "linesight\config_files\config.py" "linesight\config_files\config_copy.py"
   ```

**Run the test:**

1. **Launch TMNF via TMLoader** (not `TmForeverLauncher.exe` directly — TMLoader is what injects TMInterface)
2. Load any map and bring the car to the start line (a custom track or built-in is fine)
3. In a PowerShell terminal:
   ```powershell
   cd C:\Users\Wings\Documents\KarrarPrivate\ProjectApex-Linesight
   .\.venv\Scripts\python.exe verify_tminterface.py
   ```
4. When the script prints `Registered with TMInterface OK.`, alt-tab into TMNF and **press ↑ to start driving**. The plugin only sends `RUN_STEP_SYNC` messages while `RaceTime` is ticking; a stationary car at the start line produces no frames.

Expected output:
```
[verify] Connecting to TMInterface on 127.0.0.1:8478 ...
Connected
[verify] Registered with TMInterface OK.
[verify] Reading 5 sync messages ...
  ON_CONNECT_SYNC
  RUN_STEP_SYNC  t=<ms>
  ... (5 frames)
[verify] PASS: received 5 sync frame(s). Stack is wired up correctly.
```

If the script registers but no `RUN_STEP_SYNC` arrives, the car isn't moving — start driving. If you see `connection refused`, TMInterface didn't load the plugin (game launched without TMLoader, or `custom_port` not set).

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| TMLoader doesn't detect TMNF | Non-default install path | Re-run TMLoader installer, manually point at your TMNF folder |
| `import linesight` fails | Wrong Python version / venv not activated | `python --version` → should be 3.11; `which python` should be inside your venv |
| `torch.cuda.is_available()` returns False | Wrong PyTorch wheel | Reinstall with the `cu128` index URL (Step 5) |
| GPU op crashes with "no kernel image available" or warning about `sm_120 is not compatible` | Installed cu121 (or older) wheel on an RTX 50-series card | Uninstall torch/torchvision/torchaudio and reinstall from `https://download.pytorch.org/whl/cu128` |
| TMInterface not loading | Game launched without TMLoader | Always launch via `TmForeverLauncher.exe`, never the raw game exe |
| Verification script can't connect | Wrong port / TMInterface not running | Confirm port 8478 in user_config.py matches TMInterface's port; check TMInterface is loaded (see overlay/console message) |

---

## Optional — install the Project Apex Control Dashboard

After the training stack is working, install the dashboard for a single-pane browser UI that handles start/stop/tweak/snapshot/race-the-agent without terminal commands.

```powershell
cd C:\Users\Wings\Documents\KarrarPrivate\ProjectApex-Linesight
.\.venv\Scripts\activate
pip install -r requirements-dashboard.txt
```

Launch with `streamlit run dashboard.py`. Full manual: [docs/DASHBOARD.md](docs/DASHBOARD.md).

This step is optional — the underlying training works fine via the terminal commands documented in [QUICKSTART.md](QUICKSTART.md).

---

When everything works, continue with [QUICKSTART.md](QUICKSTART.md) to build the track and start training.
