"""
overlay_kde_equivalent_diameter_CLEAN_um.py

Clean continuous KDE overlay per visit for:
    equivalent_diameter_px

- ONLY KDE curves
- Opaque curve edge
- Very soft filled area under curve (same color)
- NO means / medians / histograms
- X axis in microns (µm) using a scale factor
- Density is adjusted to remain a proper PDF after scaling
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

FEATURE = "equivalent_diameter_px"

# --- Units handling ---
# If your CSV values are in pixels, set DATA_UNITS="px".
# If your CSV values are already in microns, set DATA_UNITS="um".
DATA_UNITS = "px"   # <-- CAMBIÁ A "um" si el CSV ya está en micras

PIXEL_SIZE_UM = 0.5  # 1 px = 0.5 µm

# KDE
BW_METHOD = 0.26     # smoothing
GRID_N = 1600

# Quantization (applies in *current data units* before scaling)
LEN_BIN = 0.0  # e.g., 0.25 if using px-quantization; in µm it would be 0.25 µm, etc.

# Figure / export
DPI = 160
EXPORT_DPI = 300
JPG_QUALITY = 95
JPG_SUBSAMPLING = 0

# Distinct colors (well separated)
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

    # scale factor from data units -> µm
    if DATA_UNITS.lower() == "px":
        scale_to_um = float(PIXEL_SIZE_UM)
    elif DATA_UNITS.lower() == "um":
        scale_to_um = 1.0
    else:
        raise ValueError("DATA_UNITS must be 'px' or 'um'")

    per_visit = []
    global_min_um = np.inf
    global_max_um = -np.inf

    for p in csvs:
        df = pd.read_csv(p)
        if FEATURE not in df.columns:
            continue

        x = robust_numeric(df[FEATURE])
        if x.size < 10:
            continue

        # Optional quantization in original units
        if LEN_BIN and LEN_BIN > 0:
            x = LEN_BIN * np.round(x / LEN_BIN)

        # Convert to µm for plotting
        x_um = x * scale_to_um

        v = infer_visit_from_name(p)
        per_visit.append((v, x_um))
        global_min_um = min(global_min_um, float(np.min(x_um)))
        global_max_um = max(global_max_um, float(np.max(x_um)))

    if not per_visit:
        raise RuntimeError("No usable data.")

    per_visit.sort(key=lambda t: visit_key(t[0]))

    # KDE grid in µm
    pad = 0.05 * (global_max_um - global_min_um)
    xmin = global_min_um - pad
    xmax = global_max_um + pad
    grid_um = np.linspace(xmin, xmax, GRID_N)

    # Plot
    fig, ax = plt.subplots(figsize=(9.4, 4.8), dpi=DPI)

    ymax = 0.0
    for i, (v, x_um) in enumerate(per_visit):
        c = COLORS[i % len(COLORS)]

        # KDE computed directly in µm
        kde = gaussian_kde(x_um, bw_method=BW_METHOD)
        y_um = kde(grid_um)  # density is per µm already (correct units)

        ymax = max(ymax, float(np.max(y_um)))

        ax.fill_between(
            grid_um, 0, y_um,
            color=c,
            alpha=FILL_ALPHA,
            linewidth=0,
        )
        ax.plot(
            grid_um, y_um,
            color=c,
            linewidth=EDGE_LW,
            alpha=EDGE_ALPHA,
            label=f"{v} (N={len(x_um)})",
        )

    ax.set_title("equivalent diameter (µm)", fontsize=13, pad=10)
    ax.set_xlabel("equivalent diameter (µm)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, ymax * 1.10)

    ax.grid(True, axis="y", alpha=0.25)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.legend(frameon=True, framealpha=0.9, fontsize=9)

    fig.tight_layout()
    save_jpg(fig, OUT_DIR / f"overlay_kde_equivalent_diameter_um_clean.jpg")
    plt.close(fig)

    print("✅ Clean KDE overlay (µm) saved.")

if __name__ == "__main__":
    main()