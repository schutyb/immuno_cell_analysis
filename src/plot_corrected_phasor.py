
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import tifffile

from phasorpy.plot import PhasorPlot


# ============================================================
# CONFIG
# ============================================================

VISIT_DIR = Path("/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449/visit01/Mosaic03_4x4_FOV600_z110_32Sp/_new")

AVG_CHANNEL = 0
G1_CHANNEL = 1
S1_CHANNEL = 2

FREQUENCY_MHZ = 80.0
INTENSITY_THRESHOLD = 50.0


# ============================================================
# LOAD DATA
# ============================================================

def load_phasor_data(tiff_path):
    arr = tifffile.imread(tiff_path)

    avg = arr[AVG_CHANNEL]
    g1 = arr[G1_CHANNEL]
    s1 = arr[S1_CHANNEL]

    mask = (
        np.isfinite(g1) &
        np.isfinite(s1) &
        (avg > INTENSITY_THRESHOLD)
    )

    return g1[mask].ravel(), s1[mask].ravel()


# ============================================================
# MAIN
# ============================================================

def main():
    phasor_path = VISIT_DIR / "phasor.tif"
    phasor_corr_path = VISIT_DIR / "phasor_corr.tif"

    g_orig, s_orig = load_phasor_data(phasor_path)
    g_corr, s_corr = load_phasor_data(phasor_corr_path)

    print(g_corr.shape, s_corr.shape)

    print(f"Original points: {len(g_orig)}")
    print(f"Corrected points: {len(g_corr)}")

    # =========================
    # ORIGINAL
    # =========================
    plot1 = PhasorPlot(
        frequency=FREQUENCY_MHZ,
        title="Original phasor (g1, s1)"
    )
    plot1.hist2d(
        g_orig,
        s_orig,
        bins=256,
    )

    # =========================
    # CORRECTED
    # =========================
    plot2 = PhasorPlot(
        frequency=FREQUENCY_MHZ,
        title="Corrected phasor (g1, s1)"
    )
    plot2.hist2d(
        g_corr,
        s_corr,
        bins=256,
    )

    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()