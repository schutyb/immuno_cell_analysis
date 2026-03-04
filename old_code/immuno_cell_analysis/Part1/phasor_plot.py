#!/usr/bin/env python3
"""
Plot phasor clouds for GREEN and BLUE from a generated phasor_data_CYX.tif,
with optional median filtering applied to g and s images.

USAGE:
1) Edit PHASOR_DATA_PATH
2) python plot_phasor_green_blue.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

from scipy.ndimage import median_filter
from phasorpy.plot import PhasorPlot


# =========================
# CONFIG
# =========================
PHASOR_DATA_PATH = Path(
    "/Users/schutyb/Documents/balu_lab/immuno_cell_analysis_data/visit_02/Mosaic02_4x4_FOV600_z105_32Sp/phasor_data_CYX.tif"
)

INTENSITY_THRESHOLD = 5.0     # mask background
SUBSAMPLE_STEP = 4            # spatial subsampling
MAX_POINTS = 400_000          # cap number of plotted points

MEDIAN_FILTER_SIZE = 7        # 7x7 median filter (set None or 0 to disable)

# Save figure next to the input TIFF
SAVE_FIGURE = True
FIGURE_NAME = "phasor_plot_green_blue.png"
FIGURE_DPI = 300


# =========================
# IO
# =========================
def read_phasor_data(path: Path):
    arr = tiff.imread(str(path))  # (C,Y,X)
    labels = None

    try:
        with tiff.TiffFile(str(path)) as tf:
            desc = tf.pages[0].description
            if desc:
                d = json.loads(desc.strip())
                labels = d.get("plane_labels", None)
    except Exception:
        pass

    if arr.ndim != 3:
        raise ValueError(f"Expected CYX TIFF, got shape {arr.shape}")

    return arr.astype(np.float32), labels


def plane_index(labels, fallback_idx: int, name: str) -> int:
    if isinstance(labels, list):
        try:
            return labels.index(name)
        except ValueError:
            return fallback_idx
    return fallback_idx


# =========================
# PROCESSING
# =========================
def apply_median(g, s, size):
    if size is None or size <= 1:
        return g, s
    g_f = median_filter(g, size=size)
    s_f = median_filter(s, size=size)
    return g_f, s_f


def flatten_with_mask(g, s, mask, subsample_step, max_points):
    g2 = g[::subsample_step, ::subsample_step]
    s2 = s[::subsample_step, ::subsample_step]
    m2 = mask[::subsample_step, ::subsample_step]

    gv = g2[m2].ravel()
    sv = s2[m2].ravel()

    if max_points is not None and gv.size > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(gv.size, size=max_points, replace=False)
        gv = gv[idx]
        sv = sv[idx]

    return gv, sv


# =========================
# PLOTTING
# =========================
def plot_one(ax, g_vals, s_vals, title):
    pp = PhasorPlot(ax=ax)
    pp.hist2d(g_vals, s_vals)
    pp.semicircle()
    ax.set_title(title)


# =========================
# MAIN
# =========================
def main():
    arr, labels = read_phasor_data(PHASOR_DATA_PATH)

    # --- indices ---
    i_g_mean = plane_index(labels, 0, "green_mean")
    i_g_g    = plane_index(labels, 1, "green_g")
    i_g_s    = plane_index(labels, 2, "green_s")

    i_b_mean = plane_index(labels, 7, "blue_mean")
    i_b_g    = plane_index(labels, 8, "blue_g")
    i_b_s    = plane_index(labels, 9, "blue_s")

    # --- extract ---
    green_mean = arr[i_g_mean]
    green_g = arr[i_g_g]
    green_s = arr[i_g_s]

    blue_mean = arr[i_b_mean]
    blue_g = arr[i_b_g]
    blue_s = arr[i_b_s]

    # --- median filter ---
    green_g, green_s = apply_median(green_g, green_s, MEDIAN_FILTER_SIZE)
    blue_g,  blue_s  = apply_median(blue_g,  blue_s,  MEDIAN_FILTER_SIZE)

    # --- masks ---
    green_mask = green_mean > INTENSITY_THRESHOLD
    blue_mask  = blue_mean  > INTENSITY_THRESHOLD

    # --- flatten ---
    gG, gS = flatten_with_mask(
        green_g, green_s, green_mask,
        SUBSAMPLE_STEP, MAX_POINTS
    )
    bG, bS = flatten_with_mask(
        blue_g, blue_s, blue_mask,
        SUBSAMPLE_STEP, MAX_POINTS
    )

    # --- plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    plot_one(
        axes[0], gG, gS,
        f"GREEN phasor | median={MEDIAN_FILTER_SIZE} | N={gG.size:,}"
    )
    plot_one(
        axes[1], bG, bS,
        f"BLUE phasor | median={MEDIAN_FILTER_SIZE} | N={bG.size:,}"
    )

    # --- save figure next to TIFF ---
    if SAVE_FIGURE:
        out_dir = PHASOR_DATA_PATH.parent
        out_path = out_dir / FIGURE_NAME
        fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"✔ Saved figure: {out_path}")

    plt.show()


if __name__ == "__main__":
    main()