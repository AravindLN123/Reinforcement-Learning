"""
Project Apex Control Dashboard.

A single-operator Streamlit app for starting/stopping training, tweaking
hyperparameters live (via Linesight's config_copy hot-reload), watching
TensorBoard scalars, snapshotting weights, and playing back / racing against
the trained agent — all in one browser tab.

One-time setup (in the project venv):
    pip install -r requirements-dashboard.txt

Launch (from repo root):
    streamlit run dashboard.py

The dashboard reads all user-specific values (TMNF username, install path,
TMInterface port, run name, active map cycle) from Linesight's user_config.py
and the live config_copy.py — nothing about the operator's machine is
hardcoded here. Map look-ups (KNOWN_MAPS below, MAP_PATH_BY_SHORT in
playback_best_run.py) cover the project's bundled tracks; users training on
additional custom maps can extend either dict.

Designed to be deletable: this is the only new file (plus
requirements-dashboard.txt). `git rm dashboard.py requirements-dashboard.txt`
removes the entire feature with no Linesight code changes to revert.
"""

from __future__ import annotations

import ast
import base64
import datetime as dt
import os
import re
import shutil
import subprocess
import sys
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import streamlit as st
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Reuse Linesight's own schedule evaluators so the displayed "current value"
# matches exactly what the training loop computes.
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "linesight"))
from trackmania_rl.utilities import (  # noqa: E402
    from_exponential_schedule,
    from_linear_schedule,
    from_staircase_schedule,
)

CONFIG_PY = REPO / "linesight" / "config_files" / "config.py"
CONFIG_COPY_PY = REPO / "linesight" / "config_files" / "config_copy.py"
CONFIG_DEFAULT_PY = REPO / "linesight" / "config_files" / "config.default.py"
TRAIN_PY = REPO / "linesight" / "scripts" / "train.py"
SAVE_DIR = REPO / "linesight" / "save"
TB_DIR = REPO / "linesight" / "tensorboard"
MAPS_DIR = REPO / "linesight" / "maps"
MAPS_REGISTRY = REPO / "maps_registry.json"
GBX_TO_VCP_SCRIPT = REPO / "linesight" / "scripts" / "tools" / "gbx_to_vcp.py"
USER_CONFIG_PY = REPO / "linesight" / "config_files" / "user_config.py"

PID_FILE = REPO / ".dashboard_training.pid"

# Plateau-detection settings (auto-stop): watch eval_race_time_robust, stop if no
# new best within this many minutes of training wall-clock.
PLATEAU_METRIC = "eval_race_time_robust"
PLATEAU_PATIENCE_MINUTES = 30

# Showcase auto-snapshot thresholds.
# Early: take a snapshot after this many frames (epsilon ~0.87 → still chaotic).
# Mid: take a snapshot on the first evaluated lap completion (race_finished eval).
SHOWCASE_EARLY_FRAMES = 80_000

# Known maps available locally (npy reference lines present in linesight/maps/).
# Each tuple: (short_name, map_path_for_TMInterface, ref_line_npy)
KNOWN_MAPS: list[tuple[str, str, str]] = [
    ("monza",      '"My Challenges/Monza.Challenge.Gbx"',         "Monza_0.5m_cl.npy"),
    ("fig8",       '"My Challenges/Figure8Track.Challenge.Gbx"',  "Figure8Track_0.5m_cl.npy"),
    ("map5",       '"My Challenges/Map5.Challenge.Gbx"',          "map5_0.5m_cl.npy"),
    ("ovaltrack1", '"My Challenges/ovaltrack1.Challenge.Gbx"',    "ovaltrack1_0.5m_cl.npy"),
    ("hock",       '"ESL-Hockolicious.Challenge.Gbx"',            "ESL-Hockolicious_0.5m_cl2.npy"),
]


# ════════════════════════════════════════════════════════════════════════════
# Config parsing & rewriting
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Assignment:
    name: str
    ranges: list[tuple[int, int]]  # one or more (start, end) 1-indexed inclusive line ranges
    value_text: str                # text of the *last* assignment's RHS (best-effort display)

    @property
    def start_line(self) -> int:
        return self.ranges[0][0]

    @property
    def end_line(self) -> int:
        return self.ranges[-1][1]


def parse_assignments(source: str) -> dict[str, Assignment]:
    """Return module-level assignments by name.

    Handles both `name = expr` (ast.Assign) and `name += expr` / etc. (ast.AugAssign).
    When a name is assigned multiple times at module level (e.g. `map_cycle = []`
    followed by `map_cycle += [...]`), all ranges are collected so rewriting can
    collapse them into a single canonical assignment.
    """
    tree = ast.parse(source)
    lines = source.splitlines()
    out: dict[str, Assignment] = {}
    for node in tree.body:
        name = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        if name is None:
            continue
        start = node.lineno
        end = getattr(node, "end_lineno", start)
        block = "\n".join(lines[start - 1 : end])
        value_text = re.sub(rf"^\s*{re.escape(name)}\s*[+\-*/]?=\s*", "", block, count=1)
        if name in out:
            out[name].ranges.append((start, end))
            out[name].value_text = value_text  # last wins for display
        else:
            out[name] = Assignment(name=name, ranges=[(start, end)], value_text=value_text)
    return out


def evaluate_value(source: str, name: str) -> Any:
    """Eval a config value safely-enough by executing config.py in a sandbox.

    config.py imports inputs_list, state_normalization, user_config — we mock the
    star imports with the actual modules so all referenced names resolve.
    """
    # Easier: just import the live config_copy module if it exists, since it
    # was generated from config.py at training start and is what Linesight reads.
    # Fall back to executing config.py in a temp namespace.
    if name in _live_config_cache:
        return _live_config_cache[name]
    return None


_live_config_cache: dict[str, Any] = {}


def reload_live_config() -> dict[str, Any]:
    """Import config_copy.py fresh and cache its module-level values."""
    global _live_config_cache
    import importlib
    # Need linesight/ on sys.path so `from config_files import ...` inside config.py works.
    cfg_path = REPO / "linesight"
    if str(cfg_path) not in sys.path:
        sys.path.insert(0, str(cfg_path))
    try:
        from config_files import config_copy as cc
        importlib.reload(cc)
        _live_config_cache = {k: getattr(cc, k) for k in dir(cc) if not k.startswith("_")}
    except Exception as e:
        st.warning(f"Couldn't load config_copy.py: {e}. Falling back to config.py defaults — some live values may be stale.")
        _live_config_cache = {}
    return _live_config_cache


def _read_run_name_from_config_py() -> str | None:
    """Parse run_name directly from config.py source (fallback when config_copy.py is absent/stale)."""
    try:
        source = CONFIG_PY.read_text(encoding="utf-8")
        assigns = parse_assignments(source)
        if "run_name" in assigns:
            return ast.literal_eval(assigns["run_name"].value_text.strip())
    except Exception:
        pass
    return None


def write_value(name: str, new_value_text: str, target: Path) -> None:
    """Atomically replace the assignment `name = ...` in `target` with new RHS text.

    Atomic via write-temp + os.replace to avoid torn reads from Linesight's
    periodic importlib.reload of config_copy.
    """
    source = target.read_text(encoding="utf-8")
    assigns = parse_assignments(source)
    if name not in assigns:
        raise KeyError(f"No top-level assignment '{name}' found in {target.name}")
    a = assigns[name]
    lines = source.splitlines(keepends=True)
    newline = "\n" if not source.endswith("\r\n") else "\r\n"
    replacement = f"{name} = {new_value_text}" + (newline if not new_value_text.endswith(newline) else "")

    # Replace the first range with the new assignment; blank out any later ranges
    # (this collapses e.g. `map_cycle = []` + `map_cycle += [...]` into a single
    # canonical `map_cycle = [...]`).
    first_start, first_end = a.ranges[0]
    new_lines: list[str] = []
    new_lines.extend(lines[: first_start - 1])
    new_lines.append(replacement)
    cursor = first_end
    for rng_start, rng_end in a.ranges[1:]:
        new_lines.extend(lines[cursor : rng_start - 1])
        # skip this range entirely (it's been collapsed into the first)
        cursor = rng_end
    new_lines.extend(lines[cursor:])
    new_source = "".join(new_lines)

    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(new_source, encoding="utf-8")
    os.replace(tmp, target)


def apply_change(name: str, new_value_text: str, training_running: bool) -> str:
    """Write `name = new_value_text` to config.py and (if running) config_copy.py.

    Returns a human-readable description of where it was applied.
    """
    write_value(name, new_value_text, CONFIG_PY)
    if training_running and CONFIG_COPY_PY.exists():
        write_value(name, new_value_text, CONFIG_COPY_PY)
        return f"Applied to config.py and config_copy.py (will be picked up live within ~1 iteration)"
    return f"Applied to config.py (will apply on next Start)"


# ════════════════════════════════════════════════════════════════════════════
# Schedule helpers (flatten-from-now)
# ════════════════════════════════════════════════════════════════════════════

def flatten_schedule_from_now(
    current_schedule: list[tuple[int, float]],
    current_step: int,
    new_value: float,
    evaluator,
) -> list[tuple[int, float]]:
    """Build a new schedule that keeps the current scheduled value until `current_step`,
    then jumps to `new_value` from there onward.
    """
    current_val = evaluator(current_schedule, current_step)
    return [
        (0, current_val),
        (max(0, current_step - 1), current_val),
        (current_step, new_value),
        (10**12, new_value),
    ]


def format_schedule(sched: list[tuple[int, float]]) -> str:
    """Format a schedule list as Python source. Keep step values readable."""
    parts = []
    for step, val in sched:
        step_str = f"{int(step):_}" if abs(step) >= 1000 else str(int(step))
        parts.append(f"    ({step_str}, {val!r}),")
    return "[\n" + "\n".join(parts) + "\n]"


# ════════════════════════════════════════════════════════════════════════════
# Process control (Windows-focused: taskkill /T /F + TM cleanup)
# ════════════════════════════════════════════════════════════════════════════

def _read_pid() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
    except ValueError:
        return None
    # Check the pid is alive
    if not _pid_alive(pid):
        try:
            PID_FILE.unlink()
        except OSError:
            pass
        return None
    return pid


def _pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True,
        )
        return str(pid) in out.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def training_status() -> tuple[str, int | None]:
    pid = _read_pid()
    if pid is None:
        return "stopped", None
    return "running", pid


def _subprocess_env() -> dict[str, str]:
    """Inherit the current env but force UTF-8 stdio so Unicode prints don't crash
    in the default cp1252 console on Windows.
    """
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def start_training() -> tuple[bool, str]:
    if _read_pid() is not None:
        return False, "Training is already running."
    log_file = REPO / "dashboard_training.log"
    # Open log for append so consecutive runs accumulate (rolled by user manually if needed).
    log_fh = open(log_file, "ab")
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    proc = subprocess.Popen(
        [sys.executable, str(TRAIN_PY)],
        cwd=str(REPO / "linesight"),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
        env=_subprocess_env(),
    )
    PID_FILE.write_text(str(proc.pid))
    return True, f"Started training (pid={proc.pid}). Logs -> dashboard_training.log"


def stop_training() -> tuple[bool, str]:
    pid = _read_pid()
    if pid is None:
        return False, "No training process is running."
    msgs = []
    # 1. Kill the training process tree (handles mp child processes).
    if sys.platform == "win32":
        r = subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            capture_output=True, text=True,
        )
        msgs.append(f"taskkill: {r.stdout.strip() or r.stderr.strip()}")
        # 2. Mirror what train.py's SIGINT handler does — clean up any orphan TM instances.
        r2 = subprocess.run(
            ["taskkill", "/F", "/IM", "TmForever.exe"],
            capture_output=True, text=True,
        )
        msgs.append(f"TmForever cleanup: {r2.stdout.strip() or r2.stderr.strip()}")
    else:
        os.kill(pid, 15)
        msgs.append(f"SIGTERM sent to {pid}")
    try:
        PID_FILE.unlink()
    except OSError:
        pass
    return True, " | ".join(msgs)


# ════════════════════════════════════════════════════════════════════════════
# Metrics (TensorBoard event files)
# ════════════════════════════════════════════════════════════════════════════

def find_tb_dirs(run_name: str) -> list[Path]:
    """Return all tensorboard dirs for this run (handles suffix rotation _2, _3, ...)."""
    if not TB_DIR.exists():
        return []
    pat = re.compile(rf"^{re.escape(run_name)}(_\d+)?$")
    return sorted([d for d in TB_DIR.iterdir() if d.is_dir() and pat.match(d.name)])


@st.cache_data(ttl=5)
def load_scalar(tb_dir_str: str, tag: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Read a scalar series from a TB event dir. Cached 5s to survive Streamlit reruns."""
    ea = EventAccumulator(tb_dir_str, size_guidance={"scalars": 0})
    try:
        ea.Reload()
    except Exception:
        return None
    if tag not in ea.Tags().get("scalars", []):
        return None
    events = ea.Scalars(tag)
    if not events:
        return None
    return np.array([e.step for e in events]), np.array([e.value for e in events])


def load_scalar_across_dirs(run_name: str, tag: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Concatenate the same tag across suffix-rotated TB dirs in step order."""
    series = []
    for d in find_tb_dirs(run_name):
        s = load_scalar(str(d), tag)
        if s is not None:
            series.append(s)
    if not series:
        return None
    steps = np.concatenate([s[0] for s in series])
    values = np.concatenate([s[1] for s in series])
    order = np.argsort(steps)
    return steps[order], values[order]


def current_step_from_stats(run_name: str) -> int | None:
    stats_file = SAVE_DIR / run_name / "accumulated_stats.joblib"
    if not stats_file.exists():
        return None
    try:
        stats = joblib.load(stats_file)
    except Exception:
        return None
    return int(stats.get("cumul_number_frames_played", 0))


# ════════════════════════════════════════════════════════════════════════════
# Snapshot weights (preserve a specific point so future training can't overwrite)
# ════════════════════════════════════════════════════════════════════════════

CHECKPOINT_FILES = ["weights1.torch", "weights2.torch", "optimizer1.torch", "scaler.torch", "accumulated_stats.joblib"]

PLAYBACK_PID_FILE = REPO / ".dashboard_playback.pid"
PLAYBACK_LOG = REPO / "dashboard_playback.log"
PLAYBACK_SCRIPT = REPO / "playback_best_run.py"


def snapshot_weights(run_name: str, note: str = "") -> tuple[bool, str]:
    """Copy the current checkpoint files into a timestamped subfolder under save/{run}/snapshots/.

    The latest weights1.torch etc. are written by Linesight every iteration, so this
    snapshot freezes whatever was on disk at the moment the button was pressed.
    """
    src_dir = SAVE_DIR / run_name
    if not src_dir.exists():
        return False, f"No save dir for run '{run_name}' yet."

    step = current_step_from_stats(run_name) or 0
    step_str = f"{step/1_000_000:.2f}M" if step >= 1_000_000 else f"{step//1000}k"
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_note = re.sub(r"[^A-Za-z0-9_-]+", "_", note).strip("_")
    folder = f"{timestamp}_step{step_str}" + (f"_{safe_note}" if safe_note else "")
    dst_dir = src_dir / "snapshots" / folder
    dst_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for f in CHECKPOINT_FILES:
        src = src_dir / f
        if src.exists():
            shutil.copy2(src, dst_dir / f)
            copied.append(f)
    if not copied:
        return False, "No checkpoint files found to snapshot."
    return True, f"Snapshot saved to {dst_dir.relative_to(REPO)} ({len(copied)} files, step={step:_})"


def list_snapshots(run_name: str) -> list[tuple[str, dt.datetime, int]]:
    """Return (folder_name, mtime, size_mb) for each snapshot under save/{run}/snapshots/."""
    snap_dir = SAVE_DIR / run_name / "snapshots"
    if not snap_dir.exists():
        return []
    out = []
    for d in sorted(snap_dir.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        total = sum(f.stat().st_size for f in d.iterdir() if f.is_file())
        out.append((d.name, dt.datetime.fromtimestamp(d.stat().st_mtime), total // (1024 * 1024)))
    return out


def delete_snapshot(run_name: str, folder_name: str) -> tuple[bool, str]:
    snap_dir = SAVE_DIR / run_name / "snapshots" / folder_name
    if not snap_dir.exists():
        return False, "Snapshot folder not found."
    shutil.rmtree(snap_dir)
    return True, f"Deleted {folder_name}"


# ════════════════════════════════════════════════════════════════════════════
# Best runs + playback (race the agent inside TMNF)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class BestRun:
    map_short: str
    time_ms: int
    folder: Path
    inputs_file: Path

    @property
    def time_str(self) -> str:
        total_s, ms = divmod(self.time_ms, 1000)
        m, s = divmod(total_s, 60)
        return f"{m}:{s:02d}.{ms:03d}" if m else f"{s}.{ms:03d}s"


def list_best_runs(run_name: str) -> list[BestRun]:
    """Scan save/{run}/best_runs/ for {map}_{time_ms}/ folders containing an .inputs file."""
    base = SAVE_DIR / run_name / "best_runs"
    if not base.exists():
        return []
    runs: list[BestRun] = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        m = re.match(r"^([A-Za-z0-9]+)_(\d+)$", d.name)
        if not m:
            continue
        inputs_file = d / f"{d.name}.inputs"
        if not inputs_file.exists():
            continue
        runs.append(BestRun(
            map_short=m.group(1),
            time_ms=int(m.group(2)),
            folder=d,
            inputs_file=inputs_file,
        ))
    # Best time first per map
    runs.sort(key=lambda r: (r.map_short, r.time_ms))
    return runs


def _read_playback_pid() -> int | None:
    if not PLAYBACK_PID_FILE.exists():
        return None
    try:
        pid = int(PLAYBACK_PID_FILE.read_text().strip())
    except ValueError:
        return None
    if not _pid_alive(pid):
        try:
            PLAYBACK_PID_FILE.unlink()
        except OSError:
            pass
        return None
    return pid


def start_playback(inputs_path: Path) -> tuple[bool, str]:
    """Launch `python playback_best_run.py <inputs_path>` in the background."""
    if _read_pid() is not None:
        return False, "Stop training first — playback can't run while training is using TMNF."
    if _read_playback_pid() is not None:
        return False, "A playback is already running."
    if not PLAYBACK_SCRIPT.exists():
        return False, f"Playback script not found at {PLAYBACK_SCRIPT}."
    if not inputs_path.exists():
        return False, f"Inputs file not found: {inputs_path}"
    log_fh = open(PLAYBACK_LOG, "ab")
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    proc = subprocess.Popen(
        [sys.executable, str(PLAYBACK_SCRIPT), str(inputs_path)],
        cwd=str(REPO),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
        env=_subprocess_env(),
    )
    PLAYBACK_PID_FILE.write_text(str(proc.pid))
    return True, f"Playback started (pid={proc.pid}). Watch your TMNF window."


def playback_status() -> tuple[str, int | None]:
    pid = _read_playback_pid()
    return ("running", pid) if pid else ("idle", None)


# ════════════════════════════════════════════════════════════════════════════
# Defaults (config.default.py — frozen on first dashboard run)
# ════════════════════════════════════════════════════════════════════════════

def ensure_defaults_file() -> bool:
    """If config.default.py does not yet exist, copy the current config.py to it.

    Returns True if a fresh default was just frozen, False if one already existed.
    """
    if CONFIG_DEFAULT_PY.exists():
        return False
    shutil.copy2(CONFIG_PY, CONFIG_DEFAULT_PY)
    return True


def reset_to_defaults(also_hot_reload: bool) -> tuple[bool, str]:
    """Overwrite config.py (and optionally config_copy.py) with config.default.py."""
    if not CONFIG_DEFAULT_PY.exists():
        return False, "No config.default.py — defaults haven't been frozen yet."
    # Atomic write via temp file + os.replace.
    for target in [CONFIG_PY] + ([CONFIG_COPY_PY] if also_hot_reload and CONFIG_COPY_PY.exists() else []):
        tmp = target.with_suffix(target.suffix + ".tmp")
        shutil.copy2(CONFIG_DEFAULT_PY, tmp)
        os.replace(tmp, target)
    where = "config.py + config_copy.py (live)" if also_hot_reload else "config.py"
    return True, f"Restored defaults to {where} from config.default.py."


# ════════════════════════════════════════════════════════════════════════════
# Plateau detection (auto-stop)
# ════════════════════════════════════════════════════════════════════════════

def detect_plateau(run_name: str) -> tuple[bool, str]:
    """Return (plateau_detected, human_message) by inspecting the watched metric.

    Plateau == best-so-far value of `eval_race_time_robust` (min, since lower=better)
    has not improved for at least PLATEAU_PATIENCE_MINUTES of training wall-clock.
    Uses event wall_time, not the dashboard's clock, so it stays correct even if
    training is slow to produce eval points.
    """
    # We need wall_time per event, so call EventAccumulator directly here.
    series_pts: list[tuple[float, float]] = []  # (wall_time, value)
    for d in find_tb_dirs(run_name):
        ea = EventAccumulator(str(d), size_guidance={"scalars": 0})
        try:
            ea.Reload()
        except Exception:
            continue
        tags = ea.Tags().get("scalars", [])
        # Also accept the same metric with a map-name suffix (Linesight appends e.g. _trained_fig8).
        candidates = [PLATEAU_METRIC] + [t for t in tags if t.startswith(PLATEAU_METRIC + "_")]
        for tag in candidates:
            if tag not in tags:
                continue
            for e in ea.Scalars(tag):
                series_pts.append((e.wall_time, e.value))

    if len(series_pts) < 3:
        return False, f"Not enough {PLATEAU_METRIC} data yet ({len(series_pts)} points)."

    series_pts.sort(key=lambda p: p[0])
    best_val = float("inf")
    best_time = series_pts[0][0]
    for wt, val in series_pts:
        if val < best_val:
            best_val = val
            best_time = wt
    latest_time = series_pts[-1][0]
    minutes_since_best = (latest_time - best_time) / 60.0
    msg = f"Best {PLATEAU_METRIC}: {best_val:.4g} | {minutes_since_best:.1f} min since last improvement (patience={PLATEAU_PATIENCE_MINUTES})"
    return (minutes_since_best >= PLATEAU_PATIENCE_MINUTES), msg


def has_first_lap_completion(run_name: str) -> bool:
    """Return True once any eval_race_time_finished_* scalar exists with at least 1 point.

    This fires the moment the agent completes its first evaluated lap.
    """
    for d in find_tb_dirs(run_name):
        ea = EventAccumulator(str(d), size_guidance={"scalars": 0})
        try:
            ea.Reload()
        except Exception:
            continue
        for tag in ea.Tags().get("scalars", []):
            if tag.startswith("eval_race_time_finished") and ea.Scalars(tag):
                return True
    return False


# ════════════════════════════════════════════════════════════════════════════
# Map cycle (de)serialisation
# ════════════════════════════════════════════════════════════════════════════

def parse_active_maps_from_config(source: str) -> list[str]:
    """Best-effort: extract short_map_names from the active (non-commented)
    `repeat((...), N)` entries inside map_cycle. We only return distinct short names."""
    assigns = parse_assignments(source)
    if "map_cycle" not in assigns:
        return []
    block = assigns["map_cycle"].value_text
    found: list[str] = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = re.search(r'repeat\(\(\s*"([^"]+)"', stripped)
        if m:
            short = m.group(1)
            if short not in found:
                found.append(short)
    return found


def build_map_cycle_text(short_names: list[str]) -> str:
    """Build a Python literal for map_cycle from chosen short names.

    Uses 4:1 explo:eval ratio for each map (Linesight's default training pattern).
    """
    lines = ["["]
    for short in short_names:
        match = next((m for m in all_known_maps() if m[0] == short), None)
        if match is None:
            continue
        _, map_path, ref_line = match
        lines.append(f'    repeat(("{short}", {map_path}, "{ref_line}", True, True), 4),')
        lines.append(f'    repeat(("{short}", {map_path}, "{ref_line}", False, True), 1),')
    lines.append("]")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# Custom map registry (maps_registry.json — user-added maps, no code editing)
# ════════════════════════════════════════════════════════════════════════════

def load_custom_maps() -> list[tuple[str, str, str]]:
    """Load user-registered maps from maps_registry.json.

    Returns list of (short_name, raw_challenge_path, npy_filename).
    raw_challenge_path has NO extra quotes — they are added in all_known_maps().
    """
    if not MAPS_REGISTRY.exists():
        return []
    try:
        data = json.loads(MAPS_REGISTRY.read_text(encoding="utf-8"))
        return [(m["short"], m["challenge_path"], m["npy_file"]) for m in data.get("maps", [])]
    except Exception:
        return []


def save_custom_maps(maps: list[tuple[str, str, str]]) -> None:
    data = {"maps": [{"short": s, "challenge_path": c, "npy_file": n} for s, c, n in maps]}
    tmp = MAPS_REGISTRY.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, MAPS_REGISTRY)


def all_known_maps() -> list[tuple[str, str, str]]:
    """Return KNOWN_MAPS + user-registered custom maps, deduped by short name.

    Custom maps store a raw challenge path; this function wraps it in quotes to
    match the format expected by build_map_cycle_text (which embeds it as a Python literal).
    """
    custom = load_custom_maps()
    known_shorts = {m[0] for m in KNOWN_MAPS}
    # Wrap raw path in quotes so it becomes a valid Python string literal in the cycle.
    custom_formatted = [
        (s, f'"{c}"', n) for s, c, n in custom if s not in known_shorts
    ]
    return list(KNOWN_MAPS) + custom_formatted


def scan_unregistered_npy() -> list[Path]:
    """Return .npy files in linesight/maps/ not yet registered in any known map."""
    if not MAPS_DIR.exists():
        return []
    registered = {m[2] for m in all_known_maps()}
    return sorted(p for p in MAPS_DIR.glob("*.npy") if p.name not in registered)


def _suggest_short_name(npy_filename: str) -> str:
    """Derive a short map name from a .npy filename (e.g. MyTrack_0.5m_cl.npy → mytrack)."""
    stem = Path(npy_filename).stem
    short = re.sub(r"_\d+\.?\d*m_cl\d*$", "", stem, flags=re.IGNORECASE)
    return re.sub(r"[^A-Za-z0-9_\-]", "_", short).lower().strip("_")


def generate_npy_from_gbx(gbx_path_str: str, output_short_name: str) -> tuple[bool, str]:
    """Run gbx_to_vcp.py as a subprocess, then rename map.npy → {name}_0.5m_cl.npy."""
    gbx = Path(gbx_path_str.strip())
    if not gbx.exists():
        return False, f"GBX file not found: {gbx}"
    dest = MAPS_DIR / f"{output_short_name}_0.5m_cl.npy"
    if dest.exists():
        return False, (
            f"`{dest.name}` already exists in linesight/maps/. "
            "Choose a different short name or delete the existing file first."
        )
    result = subprocess.run(
        [sys.executable, str(GBX_TO_VCP_SCRIPT), str(gbx)],
        cwd=str(REPO / "linesight"),
        capture_output=True, text=True,
        env=_subprocess_env(), timeout=120,
    )
    map_npy = MAPS_DIR / "map.npy"
    if result.returncode != 0 or not map_npy.exists():
        err = (result.stderr or result.stdout or "").strip()
        return False, f"gbx_to_vcp.py failed (exit {result.returncode}):\n{err}"
    map_npy.rename(dest)
    return True, f"Generated {dest.name} ({dest.stat().st_size // 1024} KB)"


def _derive_challenge_path(gbx_path_str: str) -> str:
    """Compute the TMInterface challenge path from a full .gbx file path.

    Strips everything up to and including the Challenges/ folder, giving
    e.g. 'My Challenges/MyTrack.Challenge.Gbx'. Falls back to bare filename.
    """
    try:
        gbx = Path(gbx_path_str.strip())
        tm_base = _live_config_cache.get("trackmania_base_path") or Path.home() / "Documents" / "TrackMania"
        challenges_root = Path(str(tm_base)) / "Tracks" / "Challenges"
        rel = gbx.relative_to(challenges_root)
        return rel.as_posix()
    except (ValueError, Exception):
        return Path(gbx_path_str.strip()).name


# ════════════════════════════════════════════════════════════════════════════
# Settings helpers (read/write user_config.py without opening the file)
# ════════════════════════════════════════════════════════════════════════════

def read_user_config_str(name: str) -> str:
    """Return the current value of a user_config.py field as a plain string for display.

    Prefers the already-resolved value from the live config cache (Path → str, int → str).
    Falls back to literal_eval of the raw source for simple string/int fields.
    """
    val = _live_config_cache.get(name)
    if val is not None:
        return str(val)
    try:
        source = USER_CONFIG_PY.read_text(encoding="utf-8")
        assigns = parse_assignments(source)
        if name in assigns:
            return str(ast.literal_eval(assigns[name].value_text.strip()))
    except Exception:
        pass
    return ""


def write_user_config(name: str, new_value_text: str) -> None:
    """Atomically write a new value to user_config.py (reuses write_value)."""
    write_value(name, new_value_text, USER_CONFIG_PY)


def _tmloader_ok() -> bool | None:
    """Return True if the configured TMLoader path exists, False if not, None if unknown."""
    val = _live_config_cache.get("windows_TMLoader_path")
    if val is None:
        return None
    return Path(str(val)).exists()


# ════════════════════════════════════════════════════════════════════════════
# Showcase auto-snapshot state (persisted per run so browser restarts survive)
# ════════════════════════════════════════════════════════════════════════════

def _showcase_state_path(run: str) -> Path:
    return SAVE_DIR / run / ".showcase_state.json"


def load_showcase_state(run: str) -> dict[str, bool]:
    path = _showcase_state_path(run)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return {
                "early_done": bool(data.get("early_done", False)),
                "mid_done":   bool(data.get("mid_done",   False)),
                "final_done": bool(data.get("final_done", False)),
            }
        except Exception:
            pass
    return {"early_done": False, "mid_done": False, "final_done": False}


def save_showcase_state(run: str, state: dict[str, bool]) -> None:
    path = _showcase_state_path(run)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    os.replace(tmp, path)


# ════════════════════════════════════════════════════════════════════════════
# Run management — create, rename, delete, list
# ════════════════════════════════════════════════════════════════════════════

_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,63}$")


def validate_run_name(name: str) -> str | None:
    """Return None if valid, else an error message."""
    name = name.strip()
    if not name:
        return "Run name cannot be empty."
    if not _RUN_NAME_RE.match(name):
        return "Use only letters, digits, hyphens and underscores (max 64 chars). Must start with a letter or digit."
    return None


def list_all_runs() -> list[tuple[str, int]]:
    """Return (run_name, total_steps) sorted by name for every subdir of save/."""
    if not SAVE_DIR.exists():
        return []
    return [
        (d.name, current_step_from_stats(d.name) or 0)
        for d in sorted(SAVE_DIR.iterdir())
        if d.is_dir() and not d.name.startswith(".")
    ]


def create_new_run(new_name: str, map_short_names: list[str], training_running: bool) -> tuple[bool, str]:
    """Update config.py (and config_copy.py when stopped) with new run_name + map_cycle."""
    err = validate_run_name(new_name)
    if err:
        return False, err
    try:
        map_text = build_map_cycle_text(map_short_names)
        write_value("run_name",  repr(new_name), CONFIG_PY)
        write_value("map_cycle", map_text,        CONFIG_PY)
        if not training_running and CONFIG_COPY_PY.exists():
            write_value("run_name",  repr(new_name), CONFIG_COPY_PY)
            write_value("map_cycle", map_text,        CONFIG_COPY_PY)
    except Exception as exc:
        return False, str(exc)
    return True, f"Config updated — run_name='{new_name}', maps={map_short_names}"


def rename_run(old_name: str, new_name: str) -> tuple[bool, str]:
    """Rename save/ and tensorboard dirs; update config.py + config_copy.py. Training must be stopped."""
    err = validate_run_name(new_name)
    if err:
        return False, err
    if old_name == new_name:
        return False, "New name is the same as the current name."
    if (SAVE_DIR / new_name).exists():
        return False, f"A run named '{new_name}' already exists in save/."
    msgs: list[str] = []
    save_old = SAVE_DIR / old_name
    if save_old.exists():
        save_old.rename(SAVE_DIR / new_name)
        msgs.append(f"save/{old_name} → save/{new_name}")
    for d in find_tb_dirs(old_name):
        new_tb = d.parent / d.name.replace(old_name, new_name, 1)
        d.rename(new_tb)
        msgs.append(f"tensorboard/{d.name} → tensorboard/{new_tb.name}")
    try:
        write_value("run_name", repr(new_name), CONFIG_PY)
        if CONFIG_COPY_PY.exists():
            write_value("run_name", repr(new_name), CONFIG_COPY_PY)
    except Exception as exc:
        return False, "Dirs renamed but config update failed: " + str(exc)
    return True, "Renamed: " + (", ".join(msgs) if msgs else "no data dirs found") + " · config updated"


def delete_run(name: str) -> tuple[bool, str]:
    """Delete save/ dir and all tensorboard dirs for `name`."""
    msgs: list[str] = []
    save = SAVE_DIR / name
    if save.exists():
        shutil.rmtree(save)
        msgs.append(f"save/{name}")
    for d in find_tb_dirs(name):
        shutil.rmtree(d)
        msgs.append(f"tensorboard/{d.name}")
    if not msgs:
        return False, f"No data found for run '{name}'."
    return True, "Deleted: " + ", ".join(msgs)


def switch_view_run(new_name: str) -> tuple[bool, str]:
    """Write run_name to config.py + config_copy.py so dashboard switches focus."""
    try:
        write_value("run_name", repr(new_name), CONFIG_PY)
        if CONFIG_COPY_PY.exists():
            write_value("run_name", repr(new_name), CONFIG_COPY_PY)
    except Exception as exc:
        return False, str(exc)
    return True, f"Now viewing run '{new_name}'."


# ════════════════════════════════════════════════════════════════════════════
# Streamlit UI
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Project Apex Dashboard", layout="wide")

_logo_path = REPO / "assets" / "ProjectApex_Logo.png"
_logo_b64 = (
    base64.b64encode(_logo_path.read_bytes()).decode()
    if _logo_path.exists() else ""
)
if _logo_b64:
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:20px;padding-bottom:0.5rem">'
        f'<img src="data:image/png;base64,{_logo_b64}" height="80"'
        f' style="object-fit:contain;border-radius:8px;flex-shrink:0"/>'
        f'<div style="line-height:1.15">'
        f'<h1 style="margin:0;padding:0;font-size:2.2rem;font-weight:900;color:inherit">'
        f'Project Apex</h1>'
        f'<p style="margin:0;padding:0;font-size:0.85rem;color:#94a3b8;'
        f'letter-spacing:.1em;font-weight:600">CONTROL DASHBOARD</p>'
        f'</div></div>',
        unsafe_allow_html=True,
    )
else:
    st.title("Project Apex Control Dashboard")

# Freeze defaults on the very first dashboard run (no-op afterwards).
if ensure_defaults_file():
    st.toast("Froze current config.py as config.default.py (used by 'Reset to defaults').", icon="✅")

# Persistent banner state across reruns (e.g. when auto-stop fires).
if "auto_stop_message" not in st.session_state:
    st.session_state.auto_stop_message = None
if "auto_stop_enabled" not in st.session_state:
    st.session_state.auto_stop_enabled = False

# Load live config every render so displayed values reflect any live edits.
live = reload_live_config()
run_name = live.get("run_name") or _read_run_name_from_config_py() or "untitled_run"
status, pid = training_status()
current_step = current_step_from_stats(run_name) or 0

# Load showcase snapshot state from disk (persists across browser restarts; keyed per run).
_scs_key = f"_showcase_{run_name}"
if _scs_key not in st.session_state:
    st.session_state[_scs_key] = load_showcase_state(run_name)
showcase_state: dict[str, bool] = st.session_state[_scs_key]

# ── Status bar ────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
with c1:
    badge = "🟢 Running" if status == "running" else "⚪ Stopped"
    st.metric("Status", badge, f"pid {pid}" if pid else None)
with c2:
    st.metric("Run name", run_name)
with c3:
    st.metric("Frames played", f"{current_step:_}")
with c4:
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if st.button("▶ Start", disabled=(status == "running"), use_container_width=True):
            ok, msg = start_training()
            (st.success if ok else st.warning)(msg)
            time.sleep(1.0)
            st.rerun()
    with b2:
        if st.button("■ Stop", disabled=(status != "running"), use_container_width=True,
                     help="Stops training. Restarting with the same run name resumes from where you left off (weights are preserved)."):
            ok, msg = stop_training()
            (st.success if ok else st.warning)(msg)
            time.sleep(1.0)
            st.rerun()
    with b3:
        if st.button("📸 Snapshot", use_container_width=True, help="Freeze the current weights so future training can't overwrite them."):
            ok, msg = snapshot_weights(run_name)
            (st.success if ok else st.warning)(msg)
            time.sleep(0.5)
            st.rerun()
    with b4:
        if st.button("↻ Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

# Persisted banner (e.g. plateau auto-stop result).
if st.session_state.auto_stop_message:
    st.warning(st.session_state.auto_stop_message)
    if st.button("Dismiss", key="dismiss_auto_stop"):
        st.session_state.auto_stop_message = None
        st.rerun()

# TMLoader path banner — shown if the executable can't be found so the user
# knows before clicking Start, not after training silently fails.
if _tmloader_ok() is False:
    st.warning(
        f"⚠️ TMLoader not found at `{_live_config_cache.get('windows_TMLoader_path')}`. "
        "Training will fail to launch. Fix the path in the **Settings** tab."
    )

st.divider()

tab_runs, tab_params, tab_metrics, tab_snapshots, tab_race, tab_settings, tab_logs = st.tabs(
    ["Run Manager", "Hyperparameters", "Metrics", "Snapshots", "Race the agent", "Settings", "Logs"]
)


# ── Run Manager tab ───────────────────────────────────────────────────────
with tab_runs:
    st.subheader("Active Run")
    _ri1, _ri2, _ri3 = st.columns(3)
    _active_maps_display = parse_active_maps_from_config(
        CONFIG_COPY_PY.read_text(encoding="utf-8") if CONFIG_COPY_PY.exists()
        else CONFIG_PY.read_text(encoding="utf-8")
    )
    _ri1.metric("Run name", run_name)
    _ri2.metric("Frames played", f"{current_step:_}")
    _ri3.metric("Active maps", ", ".join(_active_maps_display) or "(none)")

    st.divider()

    # ─ New Training Run ────────────────────────────────────────────────────
    st.subheader("New Training Run")
    st.caption(
        "Sets a new run name and map selection in config, then optionally starts training immediately. "
        "If training is currently running it will be stopped first."
    )

    _new_name = st.text_input(
        "Run name",
        placeholder="e.g. monza_run02",
        key="rm_new_run_name",
        help="Letters, digits, hyphens and underscores only. No spaces. Must start with a letter or digit.",
    )
    _all_maps_reg = all_known_maps()
    _default_maps = [m for m in _active_maps_display if m in {x[0] for x in _all_maps_reg}]
    if not _default_maps and _all_maps_reg:
        _default_maps = [_all_maps_reg[0][0]]
    _new_maps = st.multiselect(
        "Maps to train on",
        options=[m[0] for m in _all_maps_reg],
        default=_default_maps,
        key="rm_new_run_maps",
    )

    _btn_ok = bool(_new_name.strip()) and bool(_new_maps)
    _ncol1, _ncol2 = st.columns(2)

    with _ncol1:
        if st.button(
            "💾 Save Config Only",
            disabled=not _btn_ok,
            use_container_width=True,
            help="Write the new run name and maps to config.py without starting or stopping training.",
        ):
            _err = validate_run_name(_new_name.strip())
            if _err:
                st.error(_err)
            else:
                _ok, _msg = create_new_run(_new_name.strip(), _new_maps, status == "running")
                (st.success if _ok else st.error)(_msg)
                time.sleep(0.4)
                st.rerun()

    with _ncol2:
        if st.button(
            "▶ Create & Start Training",
            disabled=not _btn_ok,
            use_container_width=True,
            type="primary",
            help="Stop any running training, apply config changes, then immediately start the new run.",
        ):
            _err = validate_run_name(_new_name.strip())
            if _err:
                st.error(_err)
            else:
                _steps_log = []
                if status == "running":
                    _ok_s, _sm = stop_training()
                    _steps_log.append(f"Stopped previous training: {_sm}")
                    time.sleep(1.5)
                _ok_c, _cm = create_new_run(_new_name.strip(), _new_maps, False)
                _steps_log.append(f"Config: {_cm}")
                if _ok_c:
                    st.session_state.auto_stop_message = None
                    st.session_state.auto_stop_enabled = False
                    _ok_t, _tm = start_training()
                    _steps_log.append(f"Training: {_tm}")
                for _s in _steps_log:
                    st.info(_s)
                time.sleep(1.0)
                st.rerun()

    st.divider()

    # ─ Rename Current Run ──────────────────────────────────────────────────
    st.subheader("Rename Current Run")
    if status == "running":
        st.warning("Stop training before renaming — Linesight is actively writing to the save directory.")
    else:
        _rcol1, _rcol2 = st.columns([3, 1])
        with _rcol1:
            _rename_input = st.text_input(
                "New run name",
                value=run_name,
                key="rm_rename_input",
            )
        with _rcol2:
            st.write("")
            if st.button(
                "✏️ Rename",
                use_container_width=True,
                disabled=(not _rename_input or _rename_input.strip() == run_name),
            ):
                _err = validate_run_name(_rename_input.strip())
                if _err:
                    st.error(_err)
                else:
                    _ok, _msg = rename_run(run_name, _rename_input.strip())
                    (st.success if _ok else st.error)(_msg)
                    time.sleep(0.5)
                    st.rerun()

    st.divider()

    # ─ All Existing Runs ───────────────────────────────────────────────────
    st.subheader("All Existing Runs")
    _all_runs = list_all_runs()
    if not _all_runs:
        st.info("No runs found in save/ yet. Create one above to get started.")
    else:
        _hc1, _hc2, _hc3, _hc4 = st.columns([3, 2, 1.5, 1.5])
        _hc1.markdown("**Run**")
        _hc2.markdown("**Frames**")
        _hc3.markdown("**Switch View**")
        _hc4.markdown("**Delete**")

        for _rname, _rsteps in _all_runs:
            _rc1, _rc2, _rc3, _rc4 = st.columns([3, 2, 1.5, 1.5])
            _is_active = (_rname == run_name)
            _rc1.write(f"**{_rname}** ← active" if _is_active else _rname)
            _rc2.write(f"{_rsteps:_}" if _rsteps else "—")

            with _rc3:
                if _is_active:
                    st.caption("current")
                else:
                    if st.button(
                        "👁 View",
                        key=f"rm_switch_{_rname}",
                        use_container_width=True,
                        disabled=(status == "running"),
                        help="Switch the dashboard to show this run's metrics and snapshots. Stop training first.",
                    ):
                        _ok, _msg = switch_view_run(_rname)
                        (st.success if _ok else st.error)(_msg)
                        time.sleep(0.3)
                        st.rerun()

            with _rc4:
                if st.button(
                    "🗑 Delete",
                    key=f"rm_del_{_rname}",
                    use_container_width=True,
                    disabled=(_is_active and status == "running"),
                    help="Permanently delete all save/ and tensorboard data for this run.",
                ):
                    st.session_state[f"_rm_confirm_{_rname}"] = True

            if st.session_state.get(f"_rm_confirm_{_rname}"):
                st.warning(
                    f"⚠️ This will permanently delete all data for **'{_rname}'**. "
                    "This cannot be undone."
                )
                _conf_in = st.text_input(
                    f"Type **{_rname}** to confirm deletion",
                    key=f"_rm_conf_in_{_rname}",
                    placeholder=_rname,
                )
                _cc1, _cc2 = st.columns(2)
                with _cc1:
                    if st.button(
                        "✓ Confirm Delete",
                        key=f"_rm_conf_del_{_rname}",
                        type="primary",
                        disabled=(_conf_in != _rname),
                    ):
                        _ok, _msg = delete_run(_rname)
                        (st.success if _ok else st.error)(_msg)
                        st.session_state.pop(f"_rm_confirm_{_rname}", None)
                        time.sleep(0.5)
                        st.rerun()
                with _cc2:
                    if st.button("✗ Cancel", key=f"_rm_cancel_{_rname}"):
                        st.session_state.pop(f"_rm_confirm_{_rname}", None)
                        st.rerun()


# ── Hyperparameters tab ───────────────────────────────────────────────────
with tab_params:
    if status == "running":
        st.info(
            "Training is running. Changes write to **config_copy.py** for live hot-reload "
            "(picked up within ~1 iteration) **and** config.py so they survive restart."
        )
    else:
        st.info("Training is stopped. Changes write to **config.py** only and apply on next Start.")

    # ── Defaults reset ─────────────────────────────────────────────────────
    rcol1, rcol2 = st.columns([3, 1])
    with rcol1:
        if CONFIG_DEFAULT_PY.exists():
            mtime = dt.datetime.fromtimestamp(CONFIG_DEFAULT_PY.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            st.caption(f"📌 Defaults frozen on {mtime} → `linesight/config_files/config.default.py`")
        else:
            st.caption("📌 No defaults file yet — will be created on next dashboard load.")
    with rcol2:
        if st.button("↺ Reset to defaults", use_container_width=True, type="secondary"):
            ok, msg = reset_to_defaults(also_hot_reload=(status == "running"))
            (st.success if ok else st.warning)(msg)
            time.sleep(0.5)
            st.rerun()

    # ── Schedules (LR, epsilon, gamma) ─────────────────────────────────────
    st.subheader("Schedules — current value + flatten-from-now")

    SCHEDULES = [
        ("lr_schedule",                "Learning rate",     from_exponential_schedule, "%.2e"),
        ("epsilon_schedule",           "Epsilon (greedy)",  from_exponential_schedule, "%.4f"),
        ("epsilon_boltzmann_schedule", "Epsilon-Boltzmann", from_exponential_schedule, "%.4f"),
        ("gamma_schedule",             "Gamma (discount)",  from_linear_schedule,      "%.5f"),
    ]
    for cfg_name, label, evaluator, fmt in SCHEDULES:
        sched = live.get(cfg_name)
        if sched is None:
            st.warning(f"{cfg_name} not found in live config.")
            continue
        cur_val = float(evaluator(sched, current_step))
        col_a, col_b, col_c = st.columns([2, 2, 1])
        with col_a:
            st.write(f"**{label}**  (`{cfg_name}`)")
            st.caption(f"Current at step {current_step:_}: `{fmt % cur_val}`")
        with col_b:
            new_val = st.number_input(
                f"New value for {label}",
                value=cur_val, format=fmt,
                key=f"new_{cfg_name}",
                label_visibility="collapsed",
            )
        with col_c:
            if st.button("Flatten from now", key=f"btn_{cfg_name}", use_container_width=True):
                new_sched = flatten_schedule_from_now(sched, current_step, float(new_val), evaluator)
                msg = apply_change(cfg_name, format_schedule(new_sched), status == "running")
                st.success(msg)
                time.sleep(0.5)
                st.rerun()

    st.divider()

    # ── Scalar rewards / engineered rewards ────────────────────────────────
    st.subheader("Reward weights")

    SCALARS = [
        ("constant_reward_per_ms",              "Constant reward per ms (time penalty)", "%.6f"),
        ("reward_per_m_advanced_along_centerline", "Reward per metre advanced",          "%.6f"),
    ]
    for cfg_name, label, fmt in SCALARS:
        cur = live.get(cfg_name)
        if cur is None:
            st.warning(f"{cfg_name} not found in live config.")
            continue
        col_a, col_b, col_c = st.columns([2, 2, 1])
        with col_a:
            st.write(f"**{label}**  (`{cfg_name}`)")
            st.caption(f"Current: `{fmt % float(cur)}`")
        with col_b:
            new_val = st.number_input(
                f"new_{cfg_name}", value=float(cur), format=fmt,
                key=f"new_{cfg_name}", label_visibility="collapsed",
            )
        with col_c:
            if st.button("Apply", key=f"btn_{cfg_name}", use_container_width=True):
                msg = apply_change(cfg_name, repr(float(new_val)), status == "running")
                st.success(msg)
                time.sleep(0.5)
                st.rerun()

    # Engineered rewards are all single-element schedules [(0, X)] by default —
    # treat them as flat scalars to keep the UI simple.
    st.write("**Engineered reward bonuses** — flat scalar at step 0 (rewrites schedule).")
    ENG_REWARDS = [
        ("engineered_speedslide_reward_schedule",     "Speedslide"),
        ("engineered_neoslide_reward_schedule",       "Neoslide"),
        ("engineered_kamikaze_reward_schedule",       "Kamikaze"),
        ("engineered_close_to_vcp_reward_schedule",   "Close-to-VCP"),
    ]
    cols = st.columns(len(ENG_REWARDS))
    for (cfg_name, label), col in zip(ENG_REWARDS, cols):
        with col:
            sched = live.get(cfg_name) or [(0, 0.0)]
            cur = float(sched[0][1])
            new_val = st.number_input(label, value=cur, format="%.4f", key=f"new_{cfg_name}")
            if st.button(f"Apply {label}", key=f"btn_{cfg_name}", use_container_width=True):
                msg = apply_change(cfg_name, format_schedule([(0, float(new_val))]), status == "running")
                st.success(msg)
                time.sleep(0.5)
                st.rerun()

    st.divider()

    # ── Game speed ─────────────────────────────────────────────────────────
    st.subheader("Game speed")
    cur_speed = int(live.get("running_speed", 80))
    col_a, col_b, col_c = st.columns([2, 2, 1])
    with col_a:
        st.write("**Running speed** (`running_speed`)")
        st.caption("1 = real-time. Maximum is 1000+.")
    with col_b:
        new_speed = st.number_input(
            "new_running_speed", value=cur_speed, min_value=1, max_value=1000,
            step=10, key="new_running_speed", label_visibility="collapsed",
        )
    with col_c:
        if st.button("Apply", key="btn_running_speed", use_container_width=True):
            msg = apply_change("running_speed", repr(int(new_speed)), status == "running")
            st.success(msg)
            time.sleep(0.5)
            st.rerun()

    st.divider()

    # ── Map selection ──────────────────────────────────────────────────────
    st.subheader("Map cycle")
    st.caption(
        "Multi-select the maps the agent should train on (4 explo + 1 eval each). "
        "Want to add a brand new track? See **docs/ADD_NEW_TRACK.md** for the full procedure."
    )
    active_now = parse_active_maps_from_config(CONFIG_COPY_PY.read_text(encoding="utf-8") if CONFIG_COPY_PY.exists() else CONFIG_PY.read_text(encoding="utf-8"))
    available = [m[0] for m in all_known_maps()]
    chosen = st.multiselect(
        "Active maps",
        options=available,
        default=[m for m in active_now if m in available] or (available[:1] if available else []),
    )
    if st.button("Apply map cycle", disabled=not chosen):
        text = build_map_cycle_text(chosen)
        msg = apply_change("map_cycle", text, status == "running")
        if status == "running":
            msg += " — collector picks up the new cycle on its next loop iteration."
        st.success(msg)
        time.sleep(0.5)
        st.rerun()

    # ── Register a new map ─────────────────────────────────────────────────
    with st.expander("🗺️ Register a new map — make custom tracks available for training"):
        st.markdown(
            "Add any TrackMania map to the training pool in two steps: "
            "**generate** its reference line (.npy) below — no terminal needed — "
            "then **register** it so it appears in all map selection menus."
        )

        # ── Step 1: Generate reference line ────────────────────────────────
        st.markdown("#### Step 1 — Generate reference line from a .gbx file")
        st.caption(
            "Paste the full path to the challenge or replay .gbx file. "
            "The file must contain at least one ghost (challenges you created in the editor always do). "
            r"Typical path: `C:\Users\YOU\Documents\TrackMania\Tracks\Challenges\My Challenges\MyTrack.Challenge.Gbx`"
        )

        _gen_c1, _gen_c2 = st.columns([3, 1])
        with _gen_c1:
            _gen_gbx = st.text_input(
                "Full path to .gbx file",
                placeholder=r"C:\Users\YOU\Documents\TrackMania\Tracks\Challenges\My Challenges\MyTrack.Challenge.Gbx",
                key="gen_gbx_path",
                help="Can be a .Challenge.Gbx or a .Replay.Gbx file.",
            )
        with _gen_c2:
            _gen_short_input = st.text_input(
                "Short name",
                value=_suggest_short_name(Path(_gen_gbx).stem) if _gen_gbx.strip() else "",
                placeholder="e.g. mytrack",
                key="gen_short_name",
                help="Identifier for the map. Auto-suggested from the filename.",
            )

        if st.button(
            "▶ Generate reference line",
            disabled=not _gen_gbx.strip() or not _gen_short_input.strip(),
            key="gen_npy_btn",
            type="primary",
            help="Runs gbx_to_vcp.py inside the project venv. Takes a few seconds.",
        ):
            _name_err = validate_run_name(_gen_short_input.strip())
            if _name_err:
                st.error(f"Short name: {_name_err}")
            else:
                with st.spinner(f"Generating reference line for '{_gen_short_input.strip()}'…"):
                    _gen_ok, _gen_msg = generate_npy_from_gbx(_gen_gbx.strip(), _gen_short_input.strip())
                if _gen_ok:
                    _derived_path = _derive_challenge_path(_gen_gbx.strip())
                    _dest_npy = f"{_gen_short_input.strip()}_0.5m_cl.npy"
                    st.session_state["_gen_short"] = _gen_short_input.strip()
                    st.session_state["_gen_challenge"] = _derived_path
                    st.session_state["_gen_npy"] = _dest_npy
                    st.success(f"✅ {_gen_msg} — registration form pre-filled below.")
                    time.sleep(0.3)
                    st.rerun()
                else:
                    st.error(_gen_msg)

        st.divider()

        # ── Step 2: Register ────────────────────────────────────────────────
        st.markdown("#### Step 2 — Register the map")

        # Auto-detect unregistered .npy files and show as hints
        _unregistered = scan_unregistered_npy()
        if _unregistered:
            st.info(
                f"**{len(_unregistered)} unregistered .npy file(s) detected** in `linesight/maps/` "
                "— fill in their challenge path below and click **Add to Registry**:"
            )
            for _p in _unregistered:
                st.caption(
                    f"• `{_p.name}` — suggested short name: **`{_suggest_short_name(_p.name)}`**"
                )
        else:
            st.caption("All .npy files in `linesight/maps/` are already registered.")

        st.markdown("**Fill in the details and click Add to Registry:**")
        _all_npy_files = sorted(p.name for p in MAPS_DIR.glob("*.npy")) if MAPS_DIR.exists() else []

        # Pre-fill from generate step if available
        _prefill_short = st.session_state.get("_gen_short", "")
        _prefill_path  = st.session_state.get("_gen_challenge", "")
        _prefill_npy   = st.session_state.get("_gen_npy", "")

        _mreg_c1, _mreg_c2, _mreg_c3 = st.columns([2, 2, 3])
        with _mreg_c1:
            _mreg_short = st.text_input(
                "Short name",
                value=_prefill_short,
                placeholder="e.g. mytrack",
                key="mr_short",
                help="Identifier used in the map cycle. Letters, digits, hyphens and underscores only.",
            )
        with _mreg_c2:
            _npy_index = _all_npy_files.index(_prefill_npy) if _prefill_npy in _all_npy_files else 0
            _mreg_npy = st.selectbox(
                "Reference line (.npy)",
                options=_all_npy_files if _all_npy_files else ["(none found in linesight/maps/)"],
                index=_npy_index,
                key="mr_npy",
                help="The .npy file generated by gbx_to_vcp.py. Must already be in linesight/maps/.",
            )
        with _mreg_c3:
            _mreg_path = st.text_input(
                "Challenge path (TMInterface)",
                value=_prefill_path,
                placeholder="e.g. My Challenges/MyTrack.Challenge.Gbx",
                key="mr_path",
                help=(
                    "Path relative to TrackMania's Challenges folder. "
                    "Do NOT include quotes — they are added automatically. "
                    "Examples: 'My Challenges/MyTrack.Challenge.Gbx' or 'MyTrack.Challenge.Gbx' "
                    "if the file is directly in the Challenges root."
                ),
            )

        if st.button(
            "+ Add to Registry",
            disabled=not _mreg_short.strip() or not _mreg_path.strip() or not _all_npy_files,
            type="primary",
            key="mr_add_btn",
        ):
            _merr = validate_run_name(_mreg_short.strip())
            if _merr:
                st.error(f"Short name: {_merr}")
            elif _mreg_short.strip() in {m[0] for m in all_known_maps()}:
                st.error(f"Short name '{_mreg_short.strip()}' is already registered.")
            else:
                _custom = load_custom_maps()
                _custom.append((_mreg_short.strip(), _mreg_path.strip(), _mreg_npy))
                save_custom_maps(_custom)
                # Clear generate-step pre-fill so form resets cleanly
                for _k in ("_gen_short", "_gen_challenge", "_gen_npy"):
                    st.session_state.pop(_k, None)
                st.success(
                    f"Registered **'{_mreg_short.strip()}'** → `{_mreg_npy}`. "
                    "It now appears in all map selection menus."
                )
                time.sleep(0.4)
                st.rerun()

        # List existing custom maps with Remove buttons
        _custom_maps_now = load_custom_maps()
        if _custom_maps_now:
            st.divider()
            st.markdown("**Registered custom maps** (from `maps_registry.json`):")
            _mh1, _mh2, _mh3, _mh4 = st.columns([2, 3, 2, 1])
            _mh1.markdown("**Short**")
            _mh2.markdown("**Challenge path**")
            _mh3.markdown("**Reference .npy**")
            _mh4.markdown("**Remove**")
            for _mi, (_ms, _mc, _mn) in enumerate(_custom_maps_now):
                _mc1, _mc2, _mc3, _mc4 = st.columns([2, 3, 2, 1])
                _mc1.write(_ms)
                _mc2.code(_mc, language=None)
                _mc3.code(_mn, language=None)
                with _mc4:
                    if st.button("✗", key=f"mr_rm_{_mi}", help=f"Remove '{_ms}' from the registry"):
                        _custom_maps_now.pop(_mi)
                        save_custom_maps(_custom_maps_now)
                        time.sleep(0.2)
                        st.rerun()

    st.divider()

    # ── Advanced (deep internals) ──────────────────────────────────────────
    with st.expander("Advanced — deep internals (use with care)"):
        st.warning(
            "Changing IQN sampling counts (iqn_n, iqn_k) mid-run is **not validated**. "
            "Restart after changing these is recommended."
        )

        # Batch size
        cur_bs = int(live.get("batch_size", 512))
        new_bs = st.number_input("batch_size", value=cur_bs, step=64, min_value=64, max_value=4096)
        if st.button("Apply batch_size"):
            msg = apply_change("batch_size", str(int(new_bs)), status == "running")
            st.success(msg)
            time.sleep(0.5)
            st.rerun()

        # IQN params
        col1, col2, col3 = st.columns(3)
        with col1:
            cur_n = int(live.get("iqn_n", 8))
            new_n = st.number_input("iqn_n", value=cur_n, step=2, min_value=2, max_value=64)
            if st.button("Apply iqn_n"):
                msg = apply_change("iqn_n", str(int(new_n)), status == "running")
                st.success(msg)
                time.sleep(0.5)
                st.rerun()
        with col2:
            cur_k = int(live.get("iqn_k", 32))
            new_k = st.number_input("iqn_k", value=cur_k, step=2, min_value=2, max_value=128)
            if st.button("Apply iqn_k"):
                msg = apply_change("iqn_k", str(int(new_k)), status == "running")
                st.success(msg)
                time.sleep(0.5)
                st.rerun()
        with col3:
            cur_kappa = float(live.get("iqn_kappa", 5e-3))
            new_kappa = st.number_input("iqn_kappa", value=cur_kappa, format="%.5f")
            if st.button("Apply iqn_kappa"):
                msg = apply_change("iqn_kappa", repr(float(new_kappa)), status == "running")
                st.success(msg)
                time.sleep(0.5)
                st.rerun()

        # Memory size — flatten-from-now of memory_size_schedule
        st.write("**Memory (replay buffer) size** — flatten current schedule to a fixed (total, start_learn) tuple from now.")
        mem_sched = live.get("memory_size_schedule")
        if mem_sched is not None:
            cur_mem = from_staircase_schedule(mem_sched, current_step)
            st.caption(f"Current at step {current_step:_}: total={int(cur_mem[0]):_}, start_learn={int(cur_mem[1]):_}")
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                new_total = st.number_input("memory total", value=int(cur_mem[0]), step=10_000, min_value=10_000)
            with mc2:
                new_start = st.number_input("memory start_learn", value=int(cur_mem[1]), step=5_000, min_value=5_000)
            with mc3:
                if st.button("Apply memory schedule"):
                    new_sched = [
                        (0, (int(cur_mem[0]), int(cur_mem[1]))),
                        (max(0, current_step - 1), (int(cur_mem[0]), int(cur_mem[1]))),
                        (current_step, (int(new_total), int(new_start))),
                    ]
                    text = "[\n" + "\n".join(f"    ({s:_}, ({a:_}, {b:_}))," for s, (a, b) in new_sched) + "\n]"
                    msg = apply_change("memory_size_schedule", text, status == "running")
                    st.success(msg)
                    time.sleep(0.5)
                    st.rerun()


# ── Metrics tab ───────────────────────────────────────────────────────────
with tab_metrics:
    dirs = find_tb_dirs(run_name)
    if not dirs:
        st.warning(f"No TensorBoard dirs found for run `{run_name}` under {TB_DIR}")
    else:
        st.caption(f"Reading from: {', '.join(d.name for d in dirs)}")

        METRICS = [
            "loss",
            "eval_race_time_robust",
            "eval_race_time_finished",
            "explo_race_time_finished",
            "avg_Q",
            "single_zone_reached",
        ]
        ncols = 2
        rows = (len(METRICS) + ncols - 1) // ncols
        for r in range(rows):
            cols = st.columns(ncols)
            for c, col in enumerate(cols):
                i = r * ncols + c
                if i >= len(METRICS):
                    break
                tag = METRICS[i]
                # Tags may have map-name suffixes (e.g. _trained_fig8); try a few variants.
                series = None
                for candidate in [tag, f"{tag}_trained_fig8", f"{tag}_trained_oval1", f"{tag}_trained_monza"]:
                    s = load_scalar_across_dirs(run_name, candidate)
                    if s is not None:
                        series = s
                        used_tag = candidate
                        break
                with col:
                    st.write(f"**{tag}**")
                    if series is None:
                        st.caption("(no data yet)")
                    else:
                        steps, values = series
                        st.caption(f"tag: `{used_tag}` · {len(steps)} points · latest: {values[-1]:.4g}")
                        st.line_chart({tag: values.tolist()})

        st.caption("Charts cached for 5 seconds — click Refresh in the status bar for an immediate update.")


# ── Snapshots tab ─────────────────────────────────────────────────────────
with tab_snapshots:
    st.subheader("Frozen weight snapshots")
    st.caption(
        f"Each snapshot copies `{', '.join(CHECKPOINT_FILES)}` from "
        f"`save/{run_name}/` into a timestamped folder under `save/{run_name}/snapshots/`. "
        "Linesight's live checkpoint overwrites every iteration; snapshots don't."
    )

    note = st.text_input("Optional note (e.g. 'best-lap-43s', 'before-lr-tweak')", value="", max_chars=40)
    if st.button("📸 Take snapshot now", type="primary"):
        ok, msg = snapshot_weights(run_name, note=note)
        (st.success if ok else st.warning)(msg)
        time.sleep(0.5)
        st.rerun()

    st.divider()
    st.subheader("Showcase auto-snapshots")
    st.caption(
        f"Automatically saves three named snapshots for the presentation comparison: "
        f"**Early** (at {SHOWCASE_EARLY_FRAMES:,} frames — chaotic), "
        f"**Mid** (first evaluated lap completion), and **Final** (plateau auto-stop)."
    )
    col_e, col_m, col_f = st.columns(3)
    with col_e:
        st.metric("Early snapshot", "✅ Saved" if showcase_state["early_done"] else "⏳ Waiting")
    with col_m:
        st.metric("Mid snapshot", "✅ Saved" if showcase_state["mid_done"] else "⏳ Waiting")
    with col_f:
        st.metric("Final snapshot", "✅ Saved" if showcase_state["final_done"] else "⏳ Waiting")

    if not showcase_state["final_done"]:
        if st.button("🏁 Force Final Snapshot & Stop", disabled=(status != "running"),
                     help="Snapshot the current weights as 'showcase_final' and stop training immediately."):
            ok_snap, snap_msg = snapshot_weights(run_name, note="showcase_final")
            ok_stop, stop_msg = stop_training()
            showcase_state["final_done"] = True
            save_showcase_state(run_name, showcase_state)
            st.session_state.auto_stop_message = (
                f"🏁 Force-stopped for showcase. "
                f"Snapshot: {'OK — ' + snap_msg if ok_snap else 'failed — ' + snap_msg} | "
                f"Stop: {'OK' if ok_stop else 'failed'} — {stop_msg}"
            )
            st.rerun()

    st.divider()
    st.subheader("Plateau auto-stop")
    st.session_state.auto_stop_enabled = st.checkbox(
        f"Auto-stop training (and auto-snapshot) if `{PLATEAU_METRIC}` has not improved "
        f"in the last {PLATEAU_PATIENCE_MINUTES} minutes of training wall-clock",
        value=st.session_state.auto_stop_enabled,
    )
    st.caption(
        "Only fires while this dashboard tab is open (the checker runs on the 5s auto-refresh loop). "
        "If you close the tab and walk away, training keeps going."
    )
    if status == "running":
        _, plateau_msg = detect_plateau(run_name)
        st.caption(plateau_msg)

    st.divider()
    st.subheader("Existing snapshots")
    snaps = list_snapshots(run_name)
    if not snaps:
        st.info("No snapshots yet for this run.")
    else:
        for _snap_name, _snap_mtime, _snap_mb in snaps:
            _sc1, _sc2, _sc3, _sc4 = st.columns([4, 2, 1, 1])
            with _sc1:
                st.code(_snap_name, language=None)
            with _sc2:
                st.caption(f"{_snap_mtime.strftime('%Y-%m-%d %H:%M')} · {_snap_mb} MB")
            with _sc3:
                st.caption(f"`save/{run_name}/snapshots/{_snap_name}/`")
            with _sc4:
                _del_key = f"_del_snap_{_snap_name}"
                if not st.session_state.get(_del_key):
                    if st.button("🗑", key=f"del_snap_{_snap_name}",
                                 help="Delete this snapshot (will ask for confirmation)"):
                        st.session_state[_del_key] = True
                        st.rerun()
                else:
                    if st.button("✓", key=f"conf_snap_{_snap_name}", type="primary",
                                 help="Confirm delete"):
                        _ok, _msg = delete_snapshot(run_name, _snap_name)
                        st.session_state.pop(_del_key, None)
                        (st.success if _ok else st.error)(_msg)
                        time.sleep(0.3)
                        st.rerun()
                    if st.button("✗", key=f"cancel_snap_{_snap_name}",
                                 help="Cancel"):
                        st.session_state.pop(_del_key, None)
                        st.rerun()


# ── Race the agent tab ────────────────────────────────────────────────────
with tab_race:
    st.subheader("Watch the agent — and race against it")

    pb_status, pb_pid = playback_status()

    if status == "running":
        st.warning(
            "⚠️ Training is currently running and using TMNF. Stop training before running a playback "
            "(both processes can't share the game)."
        )

    st.markdown(
        f"""
**How to use this tab (step by step):**

1. **Stop training.**
2. **Launch TMNF via TMLoader** → select your profile → reach the main menu.
3. Pick a best run below and click **▶ Playback**.
4. In TMNF: **Editors → Load Track → select the same map as the playback.**
5. The agent drives automatically. If it appears desynced, double-check the race countdown speed.

Connection port: `{live.get('base_tmi_port', 8478)}`.

**To race against the agent yourself (you drive, agent is a ghost):**

1. After step 4 above, close TMLoader-TMNF.
2. Launch **regular TMNF from Steam** (not TMLoader — TMLoader is for training only).
3. In TMNF: **Solo → Race against replay → pick the `.Replay.Gbx` you just generated → start.**
4. The agent appears as a ghost car you can try to beat.
        """
    )

    st.divider()

    runs = list_best_runs(run_name)
    if not runs:
        st.info(f"No best runs recorded yet for run `{run_name}`. Train for a while first.")
    else:
        maps_avail = sorted({r.map_short for r in runs})
        chosen_map = st.selectbox(
            "Filter by map",
            options=["(all)"] + maps_avail,
            index=0,
        )
        runs_view = [r for r in runs if chosen_map in ("(all)", r.map_short)]
        st.caption(f"{len(runs_view)} best runs found (sorted by lap time, fastest first).")

        if pb_status == "running":
            # Distinguish two phases by reading the playback log tail:
            #   • before "Connected." line  → waiting for TMNF to be launched
            #   • after "Race running"      → agent is actively driving
            log_tail = ""
            if PLAYBACK_LOG.exists():
                try:
                    log_tail = PLAYBACK_LOG.read_text(encoding="utf-8", errors="replace")[-2000:]
                except OSError:
                    log_tail = ""

            if "Race running" in log_tail or "Checkpoint" in log_tail:
                st.success(f"🏎️ Agent is driving in TMNF (pid {pb_pid}). Watch your game window!")
            elif "Map load requested" in log_tail:
                st.info(f"🗺️ Map loading in TMNF (pid {pb_pid})... the agent will start in a moment.")
            elif "Connected." in log_tail:
                st.info(f"🔌 Connected to TMInterface (pid {pb_pid}). Loading the map...")
            else:
                # The script is still in its connection retry loop — TMNF isn't up yet.
                with st.status(
                    "⏳ Playback loading — now launch the game",
                    expanded=True, state="running",
                ):
                    st.markdown(
                        "⏳ Waiting for TMNF — make sure it is running at the **main menu**.\n\n"
                        f"The playback script is connecting on port `{live.get('base_tmi_port', 8478)}`."
                    )

        # Headers
        h1, h2, h3, h4, h5 = st.columns([1, 1, 2, 2, 1.2])
        h1.markdown("**Map**")
        h2.markdown("**Lap time**")
        h3.markdown("**Folder**")
        h4.markdown("**Inputs file**")
        h5.markdown("**Action**")

        # Show up to 20 best runs to keep the UI manageable
        for r in runs_view[:20]:
            c1, c2, c3, c4, c5 = st.columns([1, 1, 2, 2, 1.2])
            c1.write(r.map_short)
            c2.write(r.time_str)
            c3.code(r.folder.relative_to(REPO).as_posix(), language=None)
            c4.code(r.inputs_file.name, language=None)
            with c5:
                if st.button(
                    "▶ Playback",
                    key=f"playback_{r.folder.name}",
                    disabled=(status == "running" or pb_status == "running"),
                    use_container_width=True,
                ):
                    ok, msg = start_playback(r.inputs_file)
                    (st.success if ok else st.warning)(msg)
                    time.sleep(0.5)
                    st.rerun()

        if len(runs_view) > 20:
            st.caption(f"(showing 20 of {len(runs_view)} — fastest only)")

        st.divider()
        # Playback log tail
        st.subheader("Playback log")
        if not PLAYBACK_LOG.exists():
            st.caption("(no playback has been run yet)")
        else:
            size = PLAYBACK_LOG.stat().st_size
            with open(PLAYBACK_LOG, "rb") as f:
                if size > 10_000:
                    f.seek(size - 10_000)
                    f.readline()
                tail = f.read().decode("utf-8", errors="replace")
            st.caption(f"Tail of {PLAYBACK_LOG.name} ({size:_} bytes total)")
            st.code(tail or "(empty)", language="text")


# ── Settings tab ──────────────────────────────────────────────────────────
with tab_settings:
    st.subheader("Machine Configuration")
    st.caption(
        "These values are stored in `linesight/config_files/user_config.py`. "
        "Changing them here updates that file directly — no Python editing needed. "
        "Changes take effect the next time training starts."
    )

    st.divider()

    # ── TMNF username ───────────────────────────────────────────────────────
    st.markdown("**TrackMania username**")
    st.caption("The in-game profile name you use in TMNF. Used for hiding personal-best replays during training/playback.")
    _cfg_user = read_user_config_str("username")
    _set_c1, _set_c2 = st.columns([3, 1])
    with _set_c1:
        _new_username = st.text_input("Username", value=_cfg_user, key="set_username", label_visibility="collapsed")
    with _set_c2:
        if st.button("Apply", key="set_username_btn", use_container_width=True,
                     disabled=not _new_username.strip() or _new_username.strip() == _cfg_user):
            write_user_config("username", repr(_new_username.strip()))
            st.success(f"Username updated to '{_new_username.strip()}'.")
            time.sleep(0.4)
            st.rerun()

    st.divider()

    # ── TMLoader path ───────────────────────────────────────────────────────
    st.markdown("**TMLoader path** (Windows only)")
    st.caption(
        "Path to `TMLoader.exe`. On a Steam install this is usually inside "
        r"`C:\Program Files (x86)\Steam\steamapps\common\TrackMania Nations Forever\`. "
        "If you installed via Epic Games or a custom location, update it here."
    )
    _cfg_tmloader = read_user_config_str("windows_TMLoader_path")
    _tmloader_exists = Path(_cfg_tmloader).exists() if _cfg_tmloader else None
    _tl_badge = "✅ Found" if _tmloader_exists else ("❌ Not found" if _tmloader_exists is False else "")
    _tl_c1, _tl_c2, _tl_c3 = st.columns([4, 1, 1])
    with _tl_c1:
        _new_tmloader = st.text_input("TMLoader path", value=_cfg_tmloader, key="set_tmloader",
                                      label_visibility="collapsed")
    with _tl_c2:
        st.write("")
        st.caption(_tl_badge)
    with _tl_c3:
        if st.button("Apply", key="set_tmloader_btn", use_container_width=True,
                     disabled=not _new_tmloader.strip() or _new_tmloader.strip() == _cfg_tmloader):
            _safe = _new_tmloader.strip().replace('"', '\\"')
            write_user_config("windows_TMLoader_path", f'Path(r"{_safe}")')
            st.success("TMLoader path updated.")
            time.sleep(0.4)
            st.rerun()

    st.divider()

    # ── TrackMania base path ────────────────────────────────────────────────
    st.markdown("**TrackMania base path**")
    st.caption(
        r"Root of your TrackMania data folder. Default: `C:\Users\YOU\Documents\TrackMania`. "
        "Used to locate challenge files, replays, and autosaves."
    )
    _cfg_tmbase = read_user_config_str("trackmania_base_path")
    _tmbase_exists = Path(_cfg_tmbase).exists() if _cfg_tmbase else None
    _tb_badge = "✅ Found" if _tmbase_exists else ("❌ Not found" if _tmbase_exists is False else "")
    _tb_c1, _tb_c2, _tb_c3 = st.columns([4, 1, 1])
    with _tb_c1:
        _new_tmbase = st.text_input("TM base path", value=_cfg_tmbase, key="set_tmbase",
                                    label_visibility="collapsed")
    with _tb_c2:
        st.write("")
        st.caption(_tb_badge)
    with _tb_c3:
        if st.button("Apply", key="set_tmbase_btn", use_container_width=True,
                     disabled=not _new_tmbase.strip() or _new_tmbase.strip() == _cfg_tmbase):
            _safe = _new_tmbase.strip().replace('"', '\\"')
            write_user_config("trackmania_base_path", f'Path(r"{_safe}")')
            st.success("TrackMania base path updated.")
            time.sleep(0.4)
            st.rerun()

    st.divider()

    # ── TMInterface port ────────────────────────────────────────────────────
    st.markdown("**TMInterface port**")
    st.caption("Port number for the first TMInterface instance. Default: 8478. Only change if that port is in use.")
    _cfg_port = read_user_config_str("base_tmi_port")
    _port_val = int(_cfg_port) if _cfg_port.isdigit() else 8478
    _port_c1, _port_c2 = st.columns([2, 1])
    with _port_c1:
        _new_port = st.number_input("Port", value=_port_val, min_value=1024, max_value=65535,
                                    step=1, key="set_port", label_visibility="collapsed")
    with _port_c2:
        if st.button("Apply", key="set_port_btn", use_container_width=True,
                     disabled=(int(_new_port) == _port_val)):
            write_user_config("base_tmi_port", str(int(_new_port)))
            st.success(f"Port updated to {int(_new_port)}.")
            time.sleep(0.4)
            st.rerun()

    st.divider()
    st.caption(
        "💡 After changing any path, click **↻ Refresh** in the status bar to reload the live config "
        "and see the ✅/❌ badges update."
    )


# ── Logs tab ──────────────────────────────────────────────────────────────
with tab_logs:
    log_file = REPO / "dashboard_training.log"
    if not log_file.exists():
        st.info("No training log yet. Start training to begin capturing stdout/stderr.")
    else:
        size = log_file.stat().st_size
        # Show last ~20KB
        with open(log_file, "rb") as f:
            if size > 20_000:
                f.seek(size - 20_000)
                f.readline()  # drop partial line
            tail = f.read().decode("utf-8", errors="replace")
        st.caption(f"Tail of {log_file.name} ({size:_} bytes total)")
        st.code(tail or "(log is empty)", language="text")


# ── Auto-refresh + plateau auto-stop ──────────────────────────────────────
# Also keep refreshing while a playback subprocess is alive so its log tail updates.
_pb_status_for_refresh, _ = playback_status()
if status == "running":
    # ── Showcase auto-snapshots (early + mid + final, state persisted to disk) ──
    if not showcase_state["early_done"] and current_step >= SHOWCASE_EARLY_FRAMES:
        ok, msg = snapshot_weights(run_name, note="showcase_early")
        showcase_state["early_done"] = True
        save_showcase_state(run_name, showcase_state)
        st.toast(f"📸 Early snapshot saved — {msg}", icon="🟠")

    if not showcase_state["mid_done"] and has_first_lap_completion(run_name):
        ok, msg = snapshot_weights(run_name, note="showcase_mid_first_lap")
        showcase_state["mid_done"] = True
        save_showcase_state(run_name, showcase_state)
        st.toast(f"📸 Mid snapshot saved (first lap completed) — {msg}", icon="🟡")

    # Check plateau condition before sleeping so the user sees the result fast.
    if st.session_state.auto_stop_enabled:
        plateau, plateau_msg = detect_plateau(run_name)
        if plateau:
            ok_snap, snap_msg = snapshot_weights(run_name, note="auto_plateau")
            ok_stop, stop_msg = stop_training()
            showcase_state["final_done"] = True
            save_showcase_state(run_name, showcase_state)
            st.session_state.auto_stop_message = (
                f"🛑 Auto-stopped on plateau. {plateau_msg} | "
                f"Snapshot: {'OK — ' + snap_msg if ok_snap else 'failed — ' + snap_msg} | "
                f"Stop: {'OK' if ok_stop else 'failed'} — {stop_msg}"
            )
            st.rerun()

    # Soft 5-second autorefresh so metrics update without manual clicks.
    time.sleep(5)
    st.rerun()
elif _pb_status_for_refresh == "running":
    # Less aggressive refresh during playback (no training to plateau-check, just log tailing).
    time.sleep(3)
    st.rerun()
