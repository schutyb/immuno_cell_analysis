#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from phasorpy.plot import PhasorPlot

from color_scales import phase_intensity_to_rgb


# =========================
# CONFIG
# =========================

PHASOR_PATH = Path(
    "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449/visit01/"
    "Mosaic03_4x4_FOV600_z110_32Sp/phasor/"
    "phasor_calibrated_green_blue_mosaic.tif"
).expanduser()

FREQUENCY_MHZ = 80.0
BIN_SIZE = 8

# Guardado
SAVE_FIGURE = True
FIG_DPI = 600

PHASE_MIN_DEG = 0.0
PHASE_MAX_DEG = 65.0
PHASE_GAMMA = 0.6

INTENSITY_GAMMA = 0.99
INTENSITY_PMIN = 1.0
INTENSITY_PMAX = 99.0


# =========================
# HELPERS
# =========================

def bin_nanmean_2d(img, bin_size=8):
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


def get_valid_points(g, s):
    valid = np.isfinite(g) & np.isfinite(s)
    return g[valid].ravel(), s[valid].ravel()


def plot_phasor_on_axis(ax, g, s, title, cmap):
    plot = PhasorPlot(
        ax=ax,
        frequency=FREQUENCY_MHZ,
        title=title,
    )

    plot.hist2d(
        g,
        s,
        bins=128,
        cmap=cmap,
    )

    return plot


# =========================
# MAIN
# =========================

def main():
    phasor = tiff.imread(PHASOR_PATH).astype(np.float32)

    if phasor.ndim != 3 or phasor.shape[0] < 5:
        raise ValueError(f"Esperaba shape (5, Y, X), recibí {phasor.shape}")

    dc = phasor[0]

    g_green = phasor[1]
    s_green = phasor[2]

    g_blue = phasor[3]
    s_blue = phasor[4]

    # =========================
    # BINNING
    # =========================

    g_green_bin = bin_nanmean_2d(g_green, BIN_SIZE)
    s_green_bin = bin_nanmean_2d(s_green, BIN_SIZE)

    g_blue_bin = bin_nanmean_2d(g_blue, BIN_SIZE)
    s_blue_bin = bin_nanmean_2d(s_blue, BIN_SIZE)

    gG, sG = get_valid_points(g_green_bin, s_green_bin)
    gB, sB = get_valid_points(g_blue_bin, s_blue_bin)

    print("\nGreen detector:")
    print("G min/max:", np.nanmin(gG), np.nanmax(gG))
    print("S min/max:", np.nanmin(sG), np.nanmax(sG))

    print("\nBlue detector:")
    print("G min/max:", np.nanmin(gB), np.nanmax(gB))
    print("S min/max:", np.nanmin(sB), np.nanmax(sB))

    # =========================
    # PSEUDOCOLOR
    # =========================

    pseudocolor_green = phase_intensity_to_rgb(
        g=g_green,
        s=s_green,
        intensity=dc,
        scale="reds_to_greens",
        phase_min_deg=PHASE_MIN_DEG,
        phase_max_deg=PHASE_MAX_DEG,
        phase_gamma=PHASE_GAMMA,
        intensity_gamma=INTENSITY_GAMMA,
        intensity_pmin=INTENSITY_PMIN,
        intensity_pmax=INTENSITY_PMAX,
    )

    pseudocolor_blue = phase_intensity_to_rgb(
        g=g_blue,
        s=s_blue,
        intensity=dc,
        scale="blues_to_greens",
        phase_min_deg=PHASE_MIN_DEG,
        phase_max_deg=PHASE_MAX_DEG,
        phase_gamma=PHASE_GAMMA,
        intensity_gamma=INTENSITY_GAMMA,
        intensity_pmin=INTENSITY_PMIN,
        intensity_pmax=INTENSITY_PMAX,
    )

    # =========================
    # FIGURE
    # =========================

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax_green_phasor = axes[0, 0]
    ax_blue_phasor = axes[0, 1]
    ax_green_img = axes[1, 0]
    ax_blue_img = axes[1, 1]

    plot_phasor_on_axis(
        ax_green_phasor,
        gG,
        sG,
        title="Green detector",
        cmap="Greens",
    )

    plot_phasor_on_axis(
        ax_blue_phasor,
        gB,
        sB,
        title="Blue detector",
        cmap="Blues",
    )

    ax_green_img.imshow(pseudocolor_green)
    ax_green_img.set_title("Green pseudocolor")
    ax_green_img.axis("off")

    ax_blue_img.imshow(pseudocolor_blue)
    ax_blue_img.set_title("Blue pseudocolor")
    ax_blue_img.axis("off")

    fig.suptitle(PHASOR_PATH.parent.parent.name, fontsize=14)
    fig.tight_layout()

    # =========================
    # SAVE
    # =========================

    if SAVE_FIGURE:
        out_path = PHASOR_PATH.parent / "phasor_qc_plot.png"

        fig.savefig(
            out_path,
            dpi=FIG_DPI,
            bbox_inches="tight",
            pad_inches=0.05,
        )

        print(f"\nSaved figure to:\n{out_path}")

    plt.show()


if __name__ == "__main__":
    main()