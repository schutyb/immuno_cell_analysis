#!/usr/bin/env python3
"""
Plot one GREEN autocalibrated phasor per visit using phasorpy.

Expected structure (Option A):
ROOT/
  visit_01/Mosaic*/phasor_uncalibrated_autocal_GREEN_CYX.tif
  visit_02/Mosaic*/phasor_uncalibrated_autocal_GREEN_CYX.tif
  ...

For each visit, all mosaics are combined into a single phasor cloud.

Output:
  visit_XX/phasor_autocal_GREEN.png
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile as tiff
from phasorpy.plot import PhasorPlot


# ============================================================
# CONFIG (EDIT ME)
# ============================================================
ROOT = Path("/Users/schutyb/Documents/balu_lab/immuno_cell_analysis_data/autocalibration_out")

TIF_NAME = "phasor_uncalibrated_autocal_GREEN_CYX.tif"

INTENSITY_THRESHOLD = 5.0
SUBSAMPLE_STEP = 4
MAX_POINTS = 400_000
RNG_SEED = 0

FREQUENCY_MHZ = 80.0
OUT_PNG_NAME = "phasor_autocal_GREEN.png"
# ============================================================


def read_cyx_with_labels(path: Path):
    arr = tiff.imread(str(path))
    labels = None
    try:
        with tiff.TiffFile(str(path)) as tf:
            desc = tf.pages[0].description
            if desc:
                labels = json.loads(desc.strip()).get("plane_labels", None)
    except Exception:
        labels = None

    if arr.ndim != 3:
        raise ValueError(f"{path.name}: expected CYX TIFF, got {arr.shape}")
    return arr.astype(np.float32), labels


def plane_idx(labels, fallback: int, name: str) -> int:
    if isinstance(labels, list):
        try:
            return labels.index(name)
        except ValueError:
            return fallback
    return fallback


def collect_phasor_points(tif_paths: list[Path]):
    g_all, s_all = [], []

    for p in tif_paths:
        arr, labels = read_cyx_with_labels(p)

        i_mean = plane_idx(labels, 0, "green_mean_uncal")
        i_g    = plane_idx(labels, 1, "green_g_autocal")
        i_s    = plane_idx(labels, 2, "green_s_autocal")

        mean = arr[i_mean]
        g = arr[i_g]
        s = arr[i_s]

        mask = (mean > INTENSITY_THRESHOLD) & np.isfinite(g) & np.isfinite(s)

        g2 = g[::SUBSAMPLE_STEP, ::SUBSAMPLE_STEP]
        s2 = s[::SUBSAMPLE_STEP, ::SUBSAMPLE_STEP]
        m2 = mask[::SUBSAMPLE_STEP, ::SUBSAMPLE_STEP]

        gv = g2[m2].ravel()
        sv = s2[m2].ravel()

        if gv.size:
            g_all.append(gv)
            s_all.append(sv)

    if not g_all:
        return None, None

    g_all = np.concatenate(g_all)
    s_all = np.concatenate(s_all)

    if g_all.size > MAX_POINTS:
        rng = np.random.default_rng(RNG_SEED)
        idx = rng.choice(g_all.size, size=MAX_POINTS, replace=False)
        g_all = g_all[idx]
        s_all = s_all[idx]

    return g_all, s_all


def main():
    visits = sorted(p for p in ROOT.glob("visit_*") if p.is_dir())
    if not visits:
        raise FileNotFoundError(f"No visit_* folders found under {ROOT}")

    print(f"[INFO] Found {len(visits)} visit(s)")

    for vdir in visits:
        tif_paths = sorted(vdir.rglob(TIF_NAME))
        if not tif_paths:
            print(f"[WARN] {vdir.name}: no {TIF_NAME} found. Skipping.")
            continue

        print(f"[INFO] {vdir.name}: {len(tif_paths)} mosaic(s)")

        g, s = collect_phasor_points(tif_paths)
        if g is None or g.size == 0:
            print(f"[WARN] {vdir.name}: no valid phasor points. Skipping.")
            continue

        title = f"{vdir.name} | GREEN autocalibrated phasor | N={g.size:,}"
        plot = PhasorPlot(frequency=FREQUENCY_MHZ, title=title)
        plot.hist2d(g, s)
        plot.semicircle()

        out_png = vdir / OUT_PNG_NAME
        plot.save(str(out_png), dpi=300)

        print(f"  [SAVED] {out_png}")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()