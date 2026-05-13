#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create grant-ready FLIM phase pseudocolor images and compact phasor plots
for one selected tile from a full 4x4 phasor mosaic.

Input phasor mosaic shape:
    (5, 4800, 4800)

Planes:
    0 = DC intensity
    1 = G green
    2 = S green
    3 = G blue
    4 = S blue

Outputs:
    <MOSAIC_DIR>/images_to_grant/
        Im_00012_green_phase.png
        Im_00012_blue_phase.png
        Im_00012_green_phasor_binned4.png
        Im_00012_blue_phasor_binned4.png
"""

from pathlib import Path
import sys

import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

MOSAIC_DIR = Path(
    "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449/visit01/Mosaic03_4x4_FOV600_z110_32Sp"
).expanduser()

PHASOR_MOSAIC_PATH = (
    MOSAIC_DIR / "phasor" / "phasor_raw_green_blue_mosaic_elastin_corrected.tif"
)

TILE_NUMBER = 12

N_ROWS = 4
N_COLS = 4
TILE_SIZE = 1200

OUTPUT_DIR = MOSAIC_DIR / "images_to_grant"

SAVE_DPI = 300

PHASE_MIN_DEG = 8.0
PHASE_MAX_DEG = 38.0
PHASE_GAMMA = 0.45

INTENSITY_GAMMA = 0.99
INTENSITY_PMIN = 1.0
INTENSITY_PMAX = 99.0

PHASOR_BIN_SIZE = 8
PHASOR_POINT_SIZE = 2
PHASOR_ALPHA = 0.15


# ============================================================
# IMPORT PROJECT UTILS
# ============================================================

CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[2]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.color_scales import phase_intensity_to_rgb, make_phasor_background


# ============================================================
# FUNCTIONS
# ============================================================

def get_tile_bounds(
    tile_number: int,
    tile_size: int = 1200,
    n_rows: int = 4,
    n_cols: int = 4,
):
    """
    Tile numbering:

         1   2   3   4
         5   6   7   8
         9  10  11  12
        13  14  15  16
    """
    if tile_number < 1 or tile_number > n_rows * n_cols:
        raise ValueError(
            f"TILE_NUMBER must be between 1 and {n_rows * n_cols}. "
            f"Got {tile_number}."
        )

    idx = tile_number - 1
    row = idx // n_cols
    col = idx % n_cols

    y0 = row * tile_size
    y1 = y0 + tile_size
    x0 = col * tile_size
    x1 = x0 + tile_size

    return y0, y1, x0, x1, row, col


def load_tile_from_phasor_mosaic(
    phasor_mosaic_path: Path,
    tile_number: int,
):
    if not phasor_mosaic_path.exists():
        raise FileNotFoundError(f"Phasor mosaic not found:\n{phasor_mosaic_path}")

    stack = tiff.imread(phasor_mosaic_path)

    if stack.ndim != 3 or stack.shape[0] != 5:
        raise ValueError(f"Expected shape (5, Y, X). Got {stack.shape}")

    _, height, width = stack.shape

    expected_height = N_ROWS * TILE_SIZE
    expected_width = N_COLS * TILE_SIZE

    if height != expected_height or width != expected_width:
        raise ValueError(
            f"Unexpected mosaic size: {height} x {width}. "
            f"Expected {expected_height} x {expected_width}."
        )

    y0, y1, x0, x1, row, col = get_tile_bounds(
        tile_number=tile_number,
        tile_size=TILE_SIZE,
        n_rows=N_ROWS,
        n_cols=N_COLS,
    )

    print("\nPhasor mosaic:")
    print(phasor_mosaic_path)
    print(f"Shape       : {stack.shape}")
    print(f"Tile number : {tile_number}")
    print(f"Tile row/col: row={row + 1}, col={col + 1}")
    print(f"Crop        : y={y0}:{y1}, x={x0}:{x1}")

    tile_stack = stack[:, y0:y1, x0:x1].astype(np.float32)

    intensity = tile_stack[0]
    g_green = tile_stack[1]
    s_green = tile_stack[2]
    g_blue = tile_stack[3]
    s_blue = tile_stack[4]

    return intensity, g_green, s_green, g_blue, s_blue


def save_image(rgb: np.ndarray, output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=SAVE_DPI)
    ax.imshow(rgb)
    ax.axis("off")

    fig.savefig(
        output_path,
        dpi=SAVE_DPI,
        bbox_inches="tight",
        pad_inches=0,
    )

    plt.close(fig)


def bin_2d_mean(img: np.ndarray, bin_size: int = 4) -> np.ndarray:
    """
    Spatial binning by mean using non-overlapping bin_size x bin_size blocks.
    """
    img = np.asarray(img, dtype=np.float32)

    h, w = img.shape
    h2 = (h // bin_size) * bin_size
    w2 = (w // bin_size) * bin_size

    img_crop = img[:h2, :w2]

    img_bin = img_crop.reshape(
        h2 // bin_size,
        bin_size,
        w2 // bin_size,
        bin_size,
    )

    return np.nanmean(img_bin, axis=(1, 3)).astype(np.float32)


def plot_phasor_binned(
    g: np.ndarray,
    s: np.ndarray,
    scale_name: str,
    output_path: Path,
    bin_size: int = 4,
):
    g_bin = bin_2d_mean(g, bin_size=bin_size)
    s_bin = bin_2d_mean(s, bin_size=bin_size)

    valid = np.isfinite(g_bin) & np.isfinite(s_bin)

    g_valid = g_bin[valid]
    s_valid = s_bin[valid]

    background = make_phasor_background(
        scale=scale_name,
        phase_min_deg=PHASE_MIN_DEG,
        phase_max_deg=PHASE_MAX_DEG,
        phase_gamma=PHASE_GAMMA,
        nx=900,
        ny=600,
    )

    fig, ax = plt.subplots(figsize=(7, 5), dpi=SAVE_DPI)

    ax.imshow(
        background,
        extent=[0, 1, 0, 0.65],
        origin="lower",
        aspect="auto",
    )

    theta = np.linspace(0, np.pi, 1000)
    x = 0.5 + 0.5 * np.cos(theta)
    y = 0.5 * np.sin(theta)

    ax.plot(x, y, color="black", linewidth=1.5)

    ax.scatter(
        g_valid,
        s_valid,
        s=PHASOR_POINT_SIZE,
        c="black",
        alpha=PHASOR_ALPHA,
        linewidths=0,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.65)

    ax.set_xlabel("G")
    ax.set_ylabel("S")
    ax.set_aspect("equal")

    fig.savefig(
        output_path,
        dpi=SAVE_DPI,
        bbox_inches="tight",
        pad_inches=0.03,
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tile_name = f"Im_{TILE_NUMBER:05d}"

    intensity, g_green, s_green, g_blue, s_blue = load_tile_from_phasor_mosaic(
        phasor_mosaic_path=PHASOR_MOSAIC_PATH,
        tile_number=TILE_NUMBER,
    )

    rgb_green_phase = phase_intensity_to_rgb(
        g=g_green,
        s=s_green,
        intensity=intensity,
        scale="reds_to_greens",
        phase_min_deg=PHASE_MIN_DEG,
        phase_max_deg=PHASE_MAX_DEG,
        phase_gamma=PHASE_GAMMA,
        intensity_gamma=INTENSITY_GAMMA,
        intensity_pmin=INTENSITY_PMIN,
        intensity_pmax=INTENSITY_PMAX,
    )

    rgb_blue_phase = phase_intensity_to_rgb(
        g=g_blue,
        s=s_blue,
        intensity=intensity,
        scale="blues_to_greens",
        phase_min_deg=PHASE_MIN_DEG,
        phase_max_deg=PHASE_MAX_DEG,
        phase_gamma=PHASE_GAMMA,
        intensity_gamma=INTENSITY_GAMMA,
        intensity_pmin=INTENSITY_PMIN,
        intensity_pmax=INTENSITY_PMAX,
    )

    green_phase_out = OUTPUT_DIR / f"{tile_name}_green_phase.png"
    blue_phase_out = OUTPUT_DIR / f"{tile_name}_blue_phase.png"

    green_phasor_out = OUTPUT_DIR / f"{tile_name}_green_phasor_binned4.png"
    blue_phasor_out = OUTPUT_DIR / f"{tile_name}_blue_phasor_binned4.png"

    save_image(rgb_green_phase, green_phase_out)
    save_image(rgb_blue_phase, blue_phase_out)

    plot_phasor_binned(
        g=g_green,
        s=s_green,
        scale_name="reds_to_greens",
        output_path=green_phasor_out,
        bin_size=PHASOR_BIN_SIZE,
    )

    plot_phasor_binned(
        g=g_blue,
        s=s_blue,
        scale_name="blues_to_greens",
        output_path=blue_phasor_out,
        bin_size=PHASOR_BIN_SIZE,
    )

    print("\nSaved:")
    print(green_phase_out)
    print(blue_phase_out)
    print(green_phasor_out)
    print(blue_phasor_out)


if __name__ == "__main__":
    main()