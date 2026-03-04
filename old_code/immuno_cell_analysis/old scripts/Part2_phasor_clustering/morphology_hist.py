"""
overlay_kde_equivalent_diameter_CLEAN.py

Clean continuous KDE overlay per visit for:
    equivalent_diameter_px

- ONLY KDE curves
- Opaque curve edge
- Very soft filled area under curve (same color)
- NO means / medians / histograms
- Saves ONLY JPG
"""

from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# =========================
# CONFIG
# =========================
ROOT = Path("/Users/schutyb/Documents/balu_lab/immuno_cell_analysis_data/cells_only_out")
OUT_DIR = ROOT / "plots_overlay_kde"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE = "circularity"

# KDE
BW_METHOD = 0.26     # smoothing
GRID_N = 1600

# Quantization (0 = no quantization)
LEN_BIN = 0.0

# Figure / export
DPI = 160
EXPORT_DPI = 300
JPG_QUALITY = 95
JPG_SUBSAMPLING = 0

# Distinct colors (colorblind-friendly & well separated)
COLORS = [
    "#cfe8f3",  # azul muy claro
    "#9ecae1",  # azul claro
    "#4292c6",  # azul medio
    "#08519c",  # azul oscuro
]

# Curve appearance
EDGE_LW = 2.8
EDGE_ALPHA = 0.95
FILL_ALPHA = 0.18

# =========================
# Helpers
# =========================
def robust_numeric(series) -> np.ndarray:
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return x[np.isfinite(x)]

def infer_visit_from_name(p: Path) -> str:
    m = re.search(r"visit\s*0*([0-9]+)", p.stem, flags=re.IGNORECASE)
    if m:
        return f"visit_{int(m.group(1)):02d}"
    return p.stem

def visit_key(v: str):
    m = re.search(r"visit_(\d+)", v)
    return int(m.group(1)) if m else v

def save_jpg(fig, out_path: Path):
    fig.savefig(
        out_path,
        dpi=EXPORT_DPI,
        bbox_inches="tight",
        pil_kwargs={"quality": int(JPG_QUALITY), "subsampling": int(JPG_SUBSAMPLING)},
    )

# =========================
# Main
# =========================
def main():
    csvs = sorted(ROOT.glob("*ONLY_cells*visit*.csv"))
    if not csvs:
        raise FileNotFoundError("No visit CSVs found.")

    per_visit = []
    global_min = np.inf
    global_max = -np.inf

    for p in csvs:
        df = pd.read_csv(p)
        if FEATURE not in df.columns:
            continue

        x = robust_numeric(df[FEATURE])
        if x.size < 10:
            continue

        if LEN_BIN and LEN_BIN > 0:
            x = LEN_BIN * np.round(x / LEN_BIN)

        v = infer_visit_from_name(p)
        per_visit.append((v, x))
        global_min = min(global_min, x.min())
        global_max = max(global_max, x.max())

    if not per_visit:
        raise RuntimeError("No usable data.")

    per_visit.sort(key=lambda t: visit_key(t[0]))

    # KDE grid
    pad = 0.05 * (global_max - global_min)
    xmin = global_min - pad
    xmax = global_max + pad
    grid = np.linspace(xmin, xmax, GRID_N)

    # Plot
    fig, ax = plt.subplots(figsize=(9.4, 4.8), dpi=DPI)

    ymax = 0.0
    for i, (v, x) in enumerate(per_visit):
        c = COLORS[i % len(COLORS)]

        kde = gaussian_kde(x, bw_method=BW_METHOD)
        y = kde(grid)
        ymax = max(ymax, y.max())

        # Filled area (very soft)
        ax.fill_between(
            grid, 0, y,
            color=c,
            alpha=FILL_ALPHA,
            linewidth=0,
        )

        # Curve edge (opaque)
        ax.plot(
            grid, y,
            color=c,
            linewidth=EDGE_LW,
            alpha=EDGE_ALPHA,
            label=f"{v} (N={len(x)})",
        )

    ax.set_title(f"{FEATURE}", fontsize=13, pad=10)
    ax.set_xlabel(FEATURE, fontsize=11)
    ax.set_ylabel("Density", fontsize=11)

    ax.set_xlim(xmin, xmax)
    # ax.set_xlim(xmin, xmax/2.75)
    ax.set_ylim(0, ymax * 1.10)

    ax.grid(True, axis="y", alpha=0.25)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.legend(frameon=True, framealpha=0.9, fontsize=9)

    fig.tight_layout()
    save_jpg(fig, OUT_DIR / f"overlay_kde_{FEATURE}_clean.jpg")
    plt.close(fig)

    print("✅ Clean KDE overlay saved.")

if __name__ == "__main__":
    main()