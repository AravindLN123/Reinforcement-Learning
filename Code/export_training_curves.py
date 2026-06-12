"""
Export TensorBoard training curves as report-ready PNG figures.

Usage (from ProjectApex-Linesight/):
    python tools/export_training_curves.py

Outputs to report/Images/training_curves/
"""

from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
TB_DIR = REPO / "linesight" / "tensorboard"
OUT_DIR = REPO.parent / "report" / "Images" / "training_curves"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    "ovaltrack1": "ovaltrack1_run01",
    "figure8":    "figure8_run01",
    # Monza was added to the figure8 run's map cycle (transfer learning), so its
    # scalars live in the same TB dir as fig8 but with the _trained_monza suffix.
    # The load() helper walks the run dir + any suffix-rotated dirs (e.g. _2).
    "monza":      "figure8_run01",
}

COLORS = {
    "ovaltrack1": "#2196F3",   # blue
    "figure8":    "#F44336",   # red
    "monza":      "#4CAF50",   # green
}

plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":      150,
})


def load(run_dir: Path, tag: str):
    """Read a scalar series from a TB run directory, automatically picking up
    suffix-rotated dirs (Linesight rotates to {name}_2, _3, ... past certain
    step thresholds — see config.tensorboard_suffix_schedule).
    """
    import re
    if not run_dir.exists():
        return None, None
    base_name = run_dir.name
    parent = run_dir.parent
    pat = re.compile(rf"^{re.escape(base_name)}(_\d+)?$")
    candidate_dirs = sorted([d for d in parent.iterdir() if d.is_dir() and pat.match(d.name)])
    all_steps, all_values = [], []
    for d in candidate_dirs:
        ea = EventAccumulator(str(d), size_guidance={"scalars": 0})
        try:
            ea.Reload()
        except Exception:
            continue
        if tag not in ea.Tags().get("scalars", []):
            continue
        events = ea.Scalars(tag)
        all_steps.extend(e.step for e in events)
        all_values.extend(e.value for e in events)
    if not all_steps:
        return None, None
    steps = np.array(all_steps)
    values = np.array(all_values)
    order = np.argsort(steps)
    return steps[order], values[order]


def smooth(values, weight=0.85):
    smoothed = []
    last = values[0]
    for v in values:
        last = last * weight + v * (1 - weight)
        smoothed.append(last)
    return np.array(smoothed)


def ms_to_s(values):
    return values / 1000.0


# ── Figure 1: Figure-8 training overview (4-panel) ─────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Figure8Track — Linesight/IQN Training Overview", fontweight="bold", y=1.01)

run_dir = TB_DIR / RUNS["figure8"]
color   = COLORS["figure8"]

panels = [
    # (ax, tag, ylabel, title, convert_fn, smooth_weight)
    (axes[0, 0], "single_zone_reached_trained_fig8",    "Zones reached",       "Track Coverage (eval)",          None,   0.0),
    (axes[0, 1], "avg_Q_trained_fig8",                  "Avg Q-value",         "Q-value (eval, greedy policy)",  None,   0.7),
    (axes[1, 0], "eval_race_time_finished_trained_fig8","Lap time (s)",        "Best Lap Time — finished runs",  ms_to_s, 0.6),
    (axes[1, 1], "loss",                                "Loss",                "Training Loss",                  None,   0.7),
]

for ax, tag, ylabel, title, convert, sw in panels:
    steps, values = load(run_dir, tag)
    if steps is None:
        ax.set_visible(False)
        continue
    if convert:
        values = convert(values)
    ax.plot(steps, values, alpha=0.25, color=color, linewidth=0.8)
    ax.plot(steps, smooth(values, sw), color=color, linewidth=1.8, label="smoothed")
    ax.set_xlabel("Training steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))

# mark the best lap (43.56 s)
ax_lap = axes[1, 0]
ax_lap.axhline(43.56, color="black", linestyle="--", linewidth=1.2, label="Best lap: 43.56 s")
ax_lap.legend(fontsize=9)

fig.tight_layout()
out = OUT_DIR / "figure8_training_overview.png"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")


# ── Figure 2: ovaltrack1 training overview (4-panel) ───────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("ovaltrack1 — Linesight/IQN Training Overview", fontweight="bold", y=1.01)

run_dir = TB_DIR / RUNS["ovaltrack1"]
color   = COLORS["ovaltrack1"]

panels = [
    (axes[0, 0], "single_zone_reached_trained_oval1",    "Zones reached",      "Track Coverage (eval)",          None,    0.0),
    (axes[0, 1], "avg_Q_trained_oval1",                  "Avg Q-value",        "Q-value (eval, greedy policy)",  None,    0.7),
    (axes[1, 0], "eval_race_time_finished_trained_oval1","Lap time (s)",       "Best Lap Time — finished runs",  ms_to_s, 0.6),
    (axes[1, 1], "loss",                                 "Loss",               "Training Loss",                  None,    0.7),
]

for ax, tag, ylabel, title, convert, sw in panels:
    steps, values = load(run_dir, tag)
    if steps is None:
        ax.set_visible(False)
        continue
    if convert:
        values = convert(values)
    ax.plot(steps, values, alpha=0.25, color=color, linewidth=0.8)
    ax.plot(steps, smooth(values, sw), color=color, linewidth=1.8, label="smoothed")
    ax.set_xlabel("Training steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))

ax_lap = axes[1, 0]
ax_lap.axhline(30.9, color="black", linestyle="--", linewidth=1.2, label="Best lap: 30.9 s")
ax_lap.legend(fontsize=9)

fig.tight_layout()
out = OUT_DIR / "ovaltrack1_training_overview.png"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")


# ── Figure 2b: Monza training overview (4-panel) ───────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Monza — Linesight/IQN Training Overview", fontweight="bold", y=1.01)

run_dir = TB_DIR / RUNS["monza"]
color   = COLORS["monza"]

panels = [
    (axes[0, 0], "single_zone_reached_trained_monza",     "Zones reached",      "Track Coverage (eval)",          None,    0.0),
    (axes[0, 1], "avg_Q_trained_monza",                   "Avg Q-value",        "Q-value (eval, greedy policy)",  None,    0.7),
    (axes[1, 0], "eval_race_time_finished_trained_monza", "Lap time (s)",       "Best Lap Time — finished runs",  ms_to_s, 0.6),
    (axes[1, 1], "loss",                                  "Loss",               "Training Loss (run-wide)",       None,    0.7),
]

for ax, tag, ylabel, title, convert, sw in panels:
    steps, values = load(run_dir, tag)
    if steps is None:
        ax.set_visible(False)
        continue
    if convert:
        values = convert(values)
    ax.plot(steps, values, alpha=0.25, color=color, linewidth=0.8)
    ax.plot(steps, smooth(values, sw), color=color, linewidth=1.8, label="smoothed")
    ax.set_xlabel("Training steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))

ax_lap = axes[1, 0]
ax_lap.axhline(87.96, color="black", linestyle="--", linewidth=1.2, label="Best lap: 1:27.96")
ax_lap.legend(fontsize=9)

fig.tight_layout()
out = OUT_DIR / "monza_training_overview.png"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")


# ── Figure 3: side-by-side lap time comparison ─────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Lap Time Improvement — All Three Tracks", fontweight="bold")

for label, run_key, tag in [
    ("ovaltrack1 (oval)",   "ovaltrack1", "eval_race_time_finished_trained_oval1"),
    ("Figure8Track (fig8)", "figure8",    "eval_race_time_finished_trained_fig8"),
    ("Monza",               "monza",      "eval_race_time_finished_trained_monza"),
]:
    run_dir = TB_DIR / RUNS[run_key]
    color   = COLORS[run_key]
    steps, values = load(run_dir, tag)
    if steps is None:
        continue
    values = ms_to_s(values)
    ax.scatter(steps, values, s=4, alpha=0.2, color=color)
    ax.plot(steps, smooth(values, 0.75), color=color, linewidth=2.0, label=label)

ax.set_xlabel("Training steps")
ax.set_ylabel("Lap time (s)")
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
ax.legend()
ax.set_title("Lower is better — only finished laps shown")

fig.tight_layout()
out = OUT_DIR / "lap_time_comparison.png"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")


# ── Figure 4: epsilon decay (exploration schedule) ─────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
fig.suptitle("Exploration Decay (ε) — Figure8Track", fontweight="bold")

run_dir = TB_DIR / RUNS["figure8"]
steps, values = load(run_dir, "epsilon")
if steps is not None:
    ax.plot(steps, values, color=COLORS["figure8"], linewidth=2.0)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("ε (epsilon)")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.set_ylim(0, 1.05)

fig.tight_layout()
out = OUT_DIR / "epsilon_decay.png"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

print(f"\nAll figures written to: {OUT_DIR}")
