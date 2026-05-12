#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QC plot for raw, coumarin-calibrated, and elastin-corrected phasors.

For each mosaic, this script reads:

    phasor/phasor_raw_green_blue_mosaic.tif
    phasor/phasor_calibrated_green_blue_mosaic.tif
    phasor/phasor_raw_green_blue_mosaic_elastin_corrected.tif

and plots phasor histograms for both detector channels:

    row 1 = raw phasor
    row 2 = coumarin-calibrated phasor
    row 3 = elastin-corrected raw phasor

    col 1 = green detector
    col 2 = blue detector

Output:
    phasor/qc_raw_calibrated_corrected_green_blue.png
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from phasorpy.plot import PhasorPlot


# ============================================================
# CONFIG
# ============================================================

PATIENT_DIR = Path(
    "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449"
).expanduser()

PHASOR_SUBDIR = "phasor"

RAW_NAME = "phasor_raw_green_blue_mosaic.tif"
CAL_NAME = "phasor_calibrated_green_blue_mosaic.tif"
CORR_NAME = "phasor_raw_green_blue_mosaic_elastin_corrected.tif"

OUTPUT_NAME = "qc_raw_calibrated_corrected_green_blue.png"

FREQUENCY_MHZ = 80.0
BIN_SIZE = 8

SAVE_FIGURE = True
SHOW_FIGURE = False
FIG_DPI = 600

# Phasor planes
G_GREEN_IDX = 1
S_GREEN_IDX = 2
G_BLUE_IDX = 3
S_BLUE_IDX = 4

# Histogram settings
HIST_BINS = 160

# Optional axis limits.
# Use None for automatic limits.
X_LIMITS = None
Y_LIMITS = None

# Example fixed limits:
# X_LIMITS = (-0.2, 1.2)
# Y_LIMITS = (-0.2, 0.8)


# ============================================================
# HELPERS
# ============================================================

def natural_key(path: Path):
    return [
        int(t) if t.isdigit() else t.lower()
        for t in re.split(r"(\d+)", path.name)
    ]


def bin_nanmean_2d(img: np.ndarray, bin_size: int = 8) -> np.ndarray:
    h, w = img.shape

    h2 = (h // bin_size) * bin_size
    w2 = (w // bin_size) * bin_size

    img = img[:h2, :w2]

    binned = img.reshape(
        h2 // bin_size,
        bin_size,
        w2 // bin_size,
        bin_size,
    )

    return np.nanmean(binned, axis=(1, 3)).astype(np.float32)


def get_valid_points(g: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(g) & np.isfinite(s)
    return g[valid].ravel(), s[valid].ravel()


def read_phasor_stack(path: Path) -> np.ndarray:
    arr = tiff.imread(path)
    arr = np.asarray(arr).squeeze()

    if arr.ndim != 3:
        raise ValueError(f"Expected CYX phasor stack, got {arr.shape} in {path}")

    if arr.shape[0] < 5:
        raise ValueError(f"Expected at least 5 planes, got {arr.shape[0]} in {path}")

    return arr.astype(np.float32, copy=False)


def extract_binned_channel_points(
    stack: np.ndarray,
    *,
    g_idx: int,
    s_idx: int,
    bin_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    g = stack[g_idx]
    s = stack[s_idx]

    g_bin = bin_nanmean_2d(g, bin_size)
    s_bin = bin_nanmean_2d(s, bin_size)

    return get_valid_points(g_bin, s_bin)


def plot_phasor_axis(
    ax,
    g: np.ndarray,
    s: np.ndarray,
    *,
    title: str,
    cmap: str,
) -> None:
    plot = PhasorPlot(
        ax=ax,
        frequency=FREQUENCY_MHZ,
        title=title,
    )

    plot.hist2d(
        g,
        s,
        bins=HIST_BINS,
        cmap=cmap,
    )

    if X_LIMITS is not None:
        ax.set_xlim(*X_LIMITS)

    if Y_LIMITS is not None:
        ax.set_ylim(*Y_LIMITS)


def print_range(label: str, g: np.ndarray, s: np.ndarray) -> None:
    if len(g) == 0 or len(s) == 0:
        print(f"{label}: no valid points")
        return

    print(
        f"{label}: "
        f"G [{np.nanmin(g):.4f}, {np.nanmax(g):.4f}] | "
        f"S [{np.nanmin(s):.4f}, {np.nanmax(s):.4f}] | "
        f"n={len(g)}"
    )


# ============================================================
# PROCESS ONE MOSAIC
# ============================================================

def process_mosaic(mosaic_dir: Path) -> None:
    phasor_dir = mosaic_dir / PHASOR_SUBDIR

    raw_path = phasor_dir / RAW_NAME
    cal_path = phasor_dir / CAL_NAME
    corr_path = phasor_dir / CORR_NAME

    if not raw_path.exists():
        print(f"[SKIP] Missing raw phasor: {raw_path}")
        return

    if not cal_path.exists():
        print(f"[SKIP] Missing calibrated phasor: {cal_path}")
        return

    if not corr_path.exists():
        print(f"[SKIP] Missing corrected phasor: {corr_path}")
        return

    raw = read_phasor_stack(raw_path)
    cal = read_phasor_stack(cal_path)
    corr = read_phasor_stack(corr_path)

    datasets = [
        ("Raw", raw),
        ("Coumarin calibrated", cal),
        ("Elastin corrected", corr),
    ]

    points = {}

    for name, stack in datasets:
        g_green, s_green = extract_binned_channel_points(
            stack,
            g_idx=G_GREEN_IDX,
            s_idx=S_GREEN_IDX,
            bin_size=BIN_SIZE,
        )

        g_blue, s_blue = extract_binned_channel_points(
            stack,
            g_idx=G_BLUE_IDX,
            s_idx=S_BLUE_IDX,
            bin_size=BIN_SIZE,
        )

        points[(name, "green")] = (g_green, s_green)
        points[(name, "blue")] = (g_blue, s_blue)

    print(f"\n[INFO] {mosaic_dir.name}")
    for name, _ in datasets:
        print_range(f"{name} | green", *points[(name, "green")])
        print_range(f"{name} | blue ", *points[(name, "blue")])

    fig, axes = plt.subplots(3, 2, figsize=(13, 16))

    for row, (name, _) in enumerate(datasets):
        g_green, s_green = points[(name, "green")]
        g_blue, s_blue = points[(name, "blue")]

        plot_phasor_axis(
            axes[row, 0],
            g_green,
            s_green,
            title=f"{name} - green detector",
            cmap="Greens",
        )

        plot_phasor_axis(
            axes[row, 1],
            g_blue,
            s_blue,
            title=f"{name} - blue detector",
            cmap="Blues",
        )

    fig.suptitle(mosaic_dir.name, fontsize=14)
    fig.tight_layout()

    if SAVE_FIGURE:
        out_path = phasor_dir / OUTPUT_NAME
        fig.savefig(
            out_path,
            dpi=FIG_DPI,
            bbox_inches="tight",
            pad_inches=0.05,
        )
        print(f"[OK] Saved QC figure: {out_path}")

    if SHOW_FIGURE:
        plt.show()

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    mosaic_dirs = sorted(
        [p for p in PATIENT_DIR.glob("visit*/Mosaic*") if p.is_dir()],
        key=natural_key,
    )

    if not mosaic_dirs:
        raise RuntimeError(f"No mosaic folders found in {PATIENT_DIR}")

    print(f"[INFO] Found {len(mosaic_dirs)} mosaic folders")

    for mosaic_dir in mosaic_dirs:
        try:
            process_mosaic(mosaic_dir)
        except Exception as e:
            print(f"[ERROR] {mosaic_dir}: {e}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()