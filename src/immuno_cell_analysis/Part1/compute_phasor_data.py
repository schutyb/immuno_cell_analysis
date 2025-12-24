#!/usr/bin/env python3
"""
Compute calibrated phasor-derived FLIM maps (green / blue) using PhasorPy.

This version supports FLIM and Coumarin TIFFs with DIFFERENT image sizes.
Calibration is performed using the Coumarin reference phasor CENTER-OF-MASS (g_ref, s_ref),
computed as intensity-weighted average over the Coumarin image.

INSTRUCTIONS:
1) Paste paths in CONFIG
2) Adjust LASER_FREQUENCY_MHZ / COUMARIN_TAU_NS if needed
3) Run:
       python compute_phasor_data.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tifffile as tiff

from phasorpy.phasor import phasor_from_signal, phasor_to_polar
from phasorpy.lifetime import phasor_calibrate, polar_to_apparent_lifetime


# =========================================================
# ===================== CONFIG ============================
# =========================================================
FLIM_TIF_PATH = Path(
    "/Users/schutyb/Documents/balu_lab/immuno_cell_analysis_data/visit_04/Mosaic07_4x4_FOV600_z150_32Sp/flim_mosaic.tif"
)

COUMARIN_TIF_PATH = Path(
    "/Users/schutyb/Documents/balu_lab/immuno_cell_analysis_data/visit_04/Mosaic07_4x4_FOV600_z150_32Sp/coumarin_FOV256_z255_32Sp.tif"
)

OUTPUT_TIF_PATH = Path(
    "/Users/schutyb/Documents/balu_lab/immuno_cell_analysis_data/visit_04/Mosaic07_4x4_FOV600_z150_32Sp/phasor_data_CYX.tif"
)

LASER_FREQUENCY_MHZ = 80.0   # MHz
COUMARIN_TAU_NS = 2.5        # ns (set to your Coumarin lifetime used for calibration)
GREEN_N = 16                 # first 16 samples => green, rest => blue
UNIT_CONVERSION = 1e-3       # MHz + ns (keep as is)
# =========================================================


# ---------------------------------------------------------
# Robust TIFF reader + axes extraction
# ---------------------------------------------------------
def _try_read_axes_metadata(tf: tiff.TiffFile) -> Optional[str]:
    axes = None
    try:
        md = getattr(tf.series[0], "metadata", None)
        if isinstance(md, dict) and "axes" in md:
            axes = md["axes"]
    except Exception:
        axes = None

    if axes is None:
        try:
            desc = tf.pages[0].description
            if desc:
                d = json.loads(desc.strip())
                if isinstance(d, dict) and "axes" in d:
                    axes = d["axes"]
        except Exception:
            pass

    return axes.upper() if isinstance(axes, str) else None


def safe_read_tiff(path: Path) -> Tuple[np.ndarray, Optional[str]]:
    """
    Read TIFF robustly. If tifffile.imread fails due to reshape metadata,
    read page-by-page and stack.
    Returns (array, axes_meta_or_None).
    """
    axes = None
    try:
        with tiff.TiffFile(str(path)) as tf:
            axes = _try_read_axes_metadata(tf)
        arr = tiff.imread(str(path))
        return np.asarray(arr), axes
    except Exception:
        with tiff.TiffFile(str(path)) as tf:
            axes = _try_read_axes_metadata(tf)
            frames = [p.asarray() for p in tf.pages]
        arr = np.stack(frames, axis=0)
        return np.asarray(arr), axes


def to_cyx_auto(arr: np.ndarray, axes: Optional[str]) -> Tuple[np.ndarray, str]:
    """
    Convert a 3D array to (C,Y,X). If axes metadata is missing or doesn't contain C,
    use heuristic: smallest dimension is assumed to be channel/sample axis.
    """
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array. Got shape={arr.shape}")

    if axes is not None:
        axes = axes.upper()
        # Prefer explicit C/Y/X
        if len(axes) == 3 and all(k in axes for k in ("Y", "X")):
            if "C" in axes:
                cyx = np.transpose(arr, (axes.index("C"), axes.index("Y"), axes.index("X")))
                return cyx.astype(np.float32), f"metadata axes={axes} -> CYX"
            # Accept T/H/S as channel-like if present
            for ch_key in ("T", "H", "S"):
                if ch_key in axes:
                    cyx = np.transpose(arr, (axes.index(ch_key), axes.index("Y"), axes.index("X")))
                    return cyx.astype(np.float32), f"metadata axes={axes} treated {ch_key} as C -> CYX"

    # Heuristic fallback
    ch_axis = int(np.argmin(arr.shape))
    cyx = np.moveaxis(arr, ch_axis, 0)
    return cyx.astype(np.float32), f"heuristic: smallest axis={ch_axis} -> C"


# ---------------------------------------------------------
# Coumarin phasor reference (single point)
# ---------------------------------------------------------
def coumarin_reference_phasor(ref_cyx: np.ndarray) -> Tuple[float, float]:
    """
    Compute (g_ref, s_ref) as intensity-weighted mean of Coumarin phasor images.
    """
    mean_r, g_r, s_r = phasor_from_signal(ref_cyx, axis=0)  # mean,g,s are (Y,X)

    w = np.asarray(mean_r, dtype=np.float64)
    w = np.clip(w, 0.0, None)
    denom = float(w.sum()) + 1e-12

    g_ref = float((np.asarray(g_r, dtype=np.float64) * w).sum() / denom)
    s_ref = float((np.asarray(s_r, dtype=np.float64) * w).sum() / denom)
    return g_ref, s_ref


# ---------------------------------------------------------
# Phasor pipeline (calibrate sample with scalar ref phasor)
# ---------------------------------------------------------
def compute_metrics_calibrated_with_scalar_ref(sample_cyx: np.ndarray, g_ref: float, s_ref: float):
    """
    Returns:
      mean, g, s, phase, modulation, tau_phase, tau_mod   (all YX)
    Calibration uses scalar (g_ref, s_ref) from Coumarin COM.
    """
    mean_s, g_s, s_s = phasor_from_signal(sample_cyx, axis=0)

    # Use scalar reference; broadcast to match sample YX
    g_ref_img = np.asarray(g_ref, dtype=np.float32)
    s_ref_img = np.asarray(s_ref, dtype=np.float32)
    mean_ref_img = np.asarray(1.0, dtype=np.float32)  # not used for scalar COM; keep as 1

    g_cal, s_cal = phasor_calibrate(
        g_s, s_s,
        mean_ref_img, g_ref_img, s_ref_img,
        frequency=LASER_FREQUENCY_MHZ,
        lifetime=COUMARIN_TAU_NS,
    )

    phase, modulation = phasor_to_polar(g_cal, s_cal)

    tau_phase, tau_mod = polar_to_apparent_lifetime(
        phase,
        modulation,
        frequency=LASER_FREQUENCY_MHZ,
        unit_conversion=UNIT_CONVERSION,
    )

    return mean_s, g_cal, s_cal, phase, modulation, tau_phase, tau_mod


def save_cyx_tiff(path: Path, stack: np.ndarray, labels: list[str], meta_extra: dict):
    meta = {
        "axes": "CYX",
        "shape": list(stack.shape),
        "plane_labels": labels,
        **meta_extra,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(
        str(path),
        stack.astype(np.float32, copy=False),
        dtype=np.float32,
        description=json.dumps(meta),
        photometric="minisblack",
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    flim_raw, flim_axes = safe_read_tiff(FLIM_TIF_PATH)
    ref_raw, ref_axes = safe_read_tiff(COUMARIN_TIF_PATH)

    flim_cyx, flim_axes_used = to_cyx_auto(flim_raw, flim_axes)
    ref_cyx, ref_axes_used = to_cyx_auto(ref_raw, ref_axes)

    C_flim = flim_cyx.shape[0]
    C_ref = ref_cyx.shape[0]
    if C_flim != C_ref:
        raise ValueError(f"Channel/sample count must match. FLIM C={C_flim} vs Coumarin C={C_ref}")

    if not (1 <= GREEN_N < C_flim):
        raise ValueError(f"Invalid GREEN_N={GREEN_N} for C={C_flim}")

    # Split groups (same split applied to reference)
    green = flim_cyx[:GREEN_N]
    blue = flim_cyx[GREEN_N:]

    green_ref = ref_cyx[:GREEN_N]
    blue_ref = ref_cyx[GREEN_N:]

    # Compute scalar reference phasor (COM) for each group
    gref_g, sref_g = coumarin_reference_phasor(green_ref)
    gref_b, sref_b = coumarin_reference_phasor(blue_ref)

    # Compute calibrated metrics
    g_mean, g_g, g_s, g_ph, g_mod, g_tp, g_tm = compute_metrics_calibrated_with_scalar_ref(green, gref_g, sref_g)
    b_mean, b_g, b_s, b_ph, b_mod, b_tp, b_tm = compute_metrics_calibrated_with_scalar_ref(blue, gref_b, sref_b)

    out_stack = np.stack(
        [
            g_mean, g_g, g_s, g_ph, g_mod, g_tp, g_tm,
            b_mean, b_g, b_s, b_ph, b_mod, b_tp, b_tm,
        ],
        axis=0
    ).astype(np.float32)

    labels = [
        "green_mean", "green_g", "green_s", "green_phase", "green_modulation",
        "green_tau_phase_ns", "green_tau_mod_ns",
        "blue_mean", "blue_g", "blue_s", "blue_phase", "blue_modulation",
        "blue_tau_phase_ns", "blue_tau_mod_ns",
    ]

    meta_extra = {
        "laser_frequency_mhz": float(LASER_FREQUENCY_MHZ),
        "coumarin_tau_ns": float(COUMARIN_TAU_NS),
        "green_samples": [0, GREEN_N - 1],
        "blue_samples": [GREEN_N, C_flim - 1],
        "source_flim": str(FLIM_TIF_PATH),
        "source_coumarin": str(COUMARIN_TIF_PATH),
        "input_flim_shape_raw": list(flim_raw.shape),
        "input_ref_shape_raw": list(ref_raw.shape),
        "input_flim_axes_meta": flim_axes,
        "input_ref_axes_meta": ref_axes,
        "flim_axes_used": flim_axes_used,
        "ref_axes_used": ref_axes_used,
        "coumarin_ref_green_gs": [gref_g, sref_g],
        "coumarin_ref_blue_gs": [gref_b, sref_b],
    }

    save_cyx_tiff(OUTPUT_TIF_PATH, out_stack, labels, meta_extra)

    print("✔ Done")
    print(f"  FLIM CYX: {flim_cyx.shape} ({flim_axes_used})")
    print(f"  REF  CYX: {ref_cyx.shape} ({ref_axes_used})")
    print(f"  Coumarin ref (green) g,s = {gref_g:.6f}, {sref_g:.6f}")
    print(f"  Coumarin ref (blue)  g,s = {gref_b:.6f}, {sref_b:.6f}")
    print(f"  Output: {OUTPUT_TIF_PATH}")
    print(f"  Output shape: {out_stack.shape} (C,Y,X)")


if __name__ == "__main__":
    main()