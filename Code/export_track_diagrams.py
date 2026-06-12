"""
Generate top-down track diagrams from centerline .npy files.

Usage (from ProjectApex-Linesight/):
    python tools/export_track_diagrams.py

Outputs to report/Images/tracks/
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

REPO    = Path(__file__).resolve().parent.parent
OUT_DIR = REPO.parent / "report" / "Images" / "tracks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRACKS = [
    {
        "name":     "ovaltrack1",
        "label":    "ovaltrack1 — Oval",
        "npy":      REPO / "linesight" / "maps" / "ovaltrack1_0.5m_cl.npy",
        "color":    "#2196F3",
        "best_lap": "30.9 s",
    },
    {
        "name":     "figure8",
        "label":    "Figure8Track — Figure-8",
        "npy":      REPO / "linesight" / "maps" / "Figure8Track_0.5m_cl.npy",
        "color":    "#F44336",
        "best_lap": "43.56 s",
    },
    {
        "name":     "monza",
        "label":    "Monza — Grand-Prix circuit",
        "npy":      REPO / "linesight" / "maps" / "Monza_0.5m_cl.npy",
        "color":    "#4CAF50",
        "best_lap": "1:27.96",
    },
]

plt.rcParams.update({
    "font.family":   "sans-serif",
    "font.size":     11,
    "figure.dpi":    150,
})

TRACK_WIDTH_M = 12   # approximate TMNF road width in metres


def draw_track(ax, pts, color, width_m=TRACK_WIDTH_M):
    x, z = pts[:, 0], pts[:, 2]

    # ── filled road ribbon ──────────────────────────────────────────────────
    # Compute perpendicular offsets at each waypoint
    dx = np.gradient(x)
    dz = np.gradient(z)
    norm = np.hypot(dx, dz)
    nx = -dz / norm   # left-pointing normal
    nz =  dx / norm

    half = width_m / 2.0
    lx, lz = x + half * nx, z + half * nz
    rx, rz = x - half * nx, z - half * nz

    # close the loop
    lx = np.append(lx, lx[0]); lz = np.append(lz, lz[0])
    rx = np.append(rx, rx[0]); rz = np.append(rz, rz[0])

    # polygon: left edge forward + right edge backward
    poly_x = np.concatenate([lx, rx[::-1]])
    poly_z = np.concatenate([lz, rz[::-1]])

    ax.fill(poly_x, poly_z, color=color, alpha=0.18, zorder=1)
    ax.plot(lx, lz, color=color, linewidth=0.8, alpha=0.5, zorder=2)
    ax.plot(rx, rz, color=color, linewidth=0.8, alpha=0.5, zorder=2)

    # ── centerline with direction gradient ─────────────────────────────────
    cx = np.append(x, x[0])
    cz = np.append(z, z[0])
    points  = np.array([cx, cz]).T.reshape(-1, 1, 2)
    segs    = np.concatenate([points[:-1], points[1:]], axis=1)
    prog    = np.linspace(0, 1, len(segs))
    lc = LineCollection(segs, array=prog, cmap="cool", linewidth=1.8, zorder=3)
    ax.add_collection(lc)

    # ── start/finish marker ────────────────────────────────────────────────
    ax.scatter(x[0], z[0], s=120, color="white", edgecolors=color,
               linewidths=2.5, zorder=5)
    ax.scatter(x[0], z[0], s=30,  color=color, zorder=6)


# ── individual figures ─────────────────────────────────────────────────────────
for track in TRACKS:
    pts = np.load(track["npy"])
    x, z = pts[:, 0], pts[:, 2]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    draw_track(ax, pts, track["color"])

    ax.set_aspect("equal")
    ax.axis("off")

    # title + stats
    fig.text(0.5, 0.97, track["label"],
             ha="center", va="top", fontsize=14, fontweight="bold",
             color="white")
    fig.text(0.5, 0.93,
             f"{len(pts)} waypoints · 0.5 m spacing · best lap {track['best_lap']}",
             ha="center", va="top", fontsize=9, color="#aaaaaa")

    # start marker legend
    dot = mpatches.Patch(color=track["color"], alpha=0.6, label="Centerline reference")
    sf  = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
                     markeredgecolor=track["color"], markersize=8, label="Start / finish",
                     linewidth=0)
    ax.legend(handles=[dot, sf], loc="lower right",
              facecolor="#2a2a3e", edgecolor="none", labelcolor="white", fontsize=9)

    out = OUT_DIR / f"{track['name']}_top_view.png"
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out}")


# ── combined figure (both tracks side by side) ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.patch.set_facecolor("#1a1a2e")
fig.suptitle("Track Layouts — Top-Down View", fontsize=14, fontweight="bold",
             color="white", y=1.01)

for ax, track in zip(axes, TRACKS):
    pts = np.load(track["npy"])
    ax.set_facecolor("#1a1a2e")
    draw_track(ax, pts, track["color"])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{track['label']}\nBest lap: {track['best_lap']}",
                 color="white", fontsize=11, pad=8)

fig.tight_layout()
out = OUT_DIR / "both_tracks_top_view.png"
fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {out}")

print(f"\nAll track diagrams written to: {OUT_DIR}")
