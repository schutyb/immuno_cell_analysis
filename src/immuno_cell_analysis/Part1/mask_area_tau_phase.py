#!/usr/bin/env python3
"""
Filter connected components (structures) in a binary mask using mean tau_phase (ns)
computed from the GREEN channel tau_phase image stored in phasor_data_CYX.tif.

Output:
- Filtered binary mask (PNG, 0/255)

Assumptions:
- Mask is binary (foreground>0).
- Structures are connected components in the mask (8-connectivity).
- tau_phase image is read from plane label "green_tau_phase_ns" if available,
  else fallback to plane index 5 (based on our phasor_data ordering).

Edit PATHS + thresholds in CONFIG and run:
  python filter_mask_by_tau_phase.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import tifffile as tiff
from PIL import Image


# =========================================================
# ===================== CONFIG ============================
# =========================================================
path = "/Users/schutyb/Documents/balu_lab/immuno_cell_analysis_data/"


PHASOR_DATA_TIF = Path(
    path + "visit_04/Mosaic07_4x4_FOV600_z150_32Sp/phasor_data_CYX.tif")

STRUCTURES_MASK_PATH = Path(
    path + "visit_04/Mosaic07_4x4_FOV600_z150_32Sp/mask_filtered_min50px.png")

OUT_MASK_PNG = Path(
    path + "visit_04/Mosaic07_4x4_FOV600_z150_32Sp/mask_filtered_area_tau_phase.png")

TAU_RANGE_NS = (0.0, 12.0)   # keep if (low < mean_tau < high)
MIN_AREA_PX = 1              # optional: ignore tiny components before tau filtering
CONNECTIVITY_8 = True        # 8-connectivity recommended
# =========================================================


# ---------------------------------------------------------
# I/O
# ---------------------------------------------------------
def load_mask_2d(path: Path) -> np.ndarray:
    """Load a 2D mask (PNG/TIF/JPG). Returns uint8 2D array."""
    ext = path.suffix.lower()
    if ext in [".tif", ".tiff"]:
        arr = tiff.imread(str(path))
    else:
        arr = np.array(Image.open(str(path)))

    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint8)


def save_mask_png(path: Path, mask01: np.ndarray) -> None:
    """Save 2D binary mask as PNG (0/255)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    m = (mask01 > 0).astype(np.uint8) * 255
    Image.fromarray(m, mode="L").save(str(path))


def read_phasor_planes(path: Path) -> Tuple[np.ndarray, Optional[List[str]]]:
    """Read phasor_data (C,Y,X) and plane labels if present."""
    arr = tiff.imread(str(path))
    labels = None

    try:
        with tiff.TiffFile(str(path)) as tf:
            desc = tf.pages[0].description
            if desc:
                d = json.loads(desc.strip())
                labels = d.get("plane_labels", None)
    except Exception:
        labels = None

    if arr.ndim != 3:
        raise ValueError(f"Expected (C,Y,X) phasor_data TIFF. Got shape={arr.shape}")

    return arr.astype(np.float32), labels


def find_plane(labels: Optional[List[str]], fallback: int, name: str) -> int:
    """Find plane index by label; if not found use fallback."""
    if isinstance(labels, list):
        try:
            return labels.index(name)
        except ValueError:
            pass
    return fallback


# ---------------------------------------------------------
# Connected components (8-connectivity)
# ---------------------------------------------------------
def connected_components(mask_bool: np.ndarray, connectivity_8: bool = True) -> Tuple[np.ndarray, List[int]]:
    """
    Label connected components in a boolean mask.
    Returns:
      labels: int32 (0 background, 1..N components)
      areas: list of areas where areas[i-1] corresponds to label i
    """
    H, W = mask_bool.shape
    labels = np.zeros((H, W), dtype=np.int32)

    if connectivity_8:
        nbrs = [(-1, -1), (-1, 0), (-1, 1),
                ( 0, -1),          ( 0, 1),
                ( 1, -1), ( 1, 0), ( 1, 1)]
    else:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    current = 0
    areas: List[int] = []

    for y in range(H):
        for x in range(W):
            if not mask_bool[y, x] or labels[y, x] != 0:
                continue

            current += 1
            stack = [(y, x)]
            labels[y, x] = current
            area = 0

            while stack:
                cy, cx = stack.pop()
                area += 1
                for dy, dx in nbrs:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        if mask_bool[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current
                            stack.append((ny, nx))

            areas.append(area)

    return labels, areas


# ---------------------------------------------------------
# Mean tau per component
# ---------------------------------------------------------
def compute_component_mean(labels: np.ndarray, value_img: np.ndarray, n_labels: int) -> np.ndarray:
    """
    Compute mean value per label (1..n_labels). Returns array length n_labels (float64).
    """
    lab = labels.ravel()
    val = value_img.ravel().astype(np.float64)

    fg = lab > 0
    lab_fg = lab[fg]
    val_fg = val[fg]

    sums = np.bincount(lab_fg, weights=val_fg, minlength=n_labels + 1)
    cnts = np.bincount(lab_fg, minlength=n_labels + 1)

    means = np.zeros(n_labels + 1, dtype=np.float64)
    ok = cnts > 0
    means[ok] = sums[ok] / cnts[ok]

    return means[1:]  # drop background


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    # --- load tau_phase (green) ---
    phasor, labels = read_phasor_planes(PHASOR_DATA_TIF)

    # Default ordering from our phasor_data:
    # 0 mean, 1 g, 2 s, 3 phase, 4 modulation, 5 tau_phase, 6 tau_mod, ...
    idx_tau = find_plane(labels, fallback=5, name="green_tau_phase_ns")
    tau_phase = phasor[idx_tau]  # (Y,X)

    # --- load structures mask ---
    mask = load_mask_2d(STRUCTURES_MASK_PATH)
    mask_bool = mask > 0

    if tau_phase.shape != mask_bool.shape:
        raise ValueError(
            f"Shape mismatch:\n"
            f"  tau_phase: {tau_phase.shape}\n"
            f"  mask:      {mask_bool.shape}\n"
            f"They must have the same YX size."
        )

    # --- label structures ---
    labels_img, areas = connected_components(mask_bool, connectivity_8=CONNECTIVITY_8)
    n = len(areas)
    if n == 0:
        raise ValueError("No structures found in mask (mask is empty?).")

    areas_arr = np.array(areas, dtype=np.int64)
    keep_area = areas_arr >= int(MIN_AREA_PX)

    # --- mean tau per structure ---
    mean_tau = compute_component_mean(labels_img, tau_phase, n)

    low, high = TAU_RANGE_NS
    keep_tau = (mean_tau > low) & (mean_tau < high)

    keep = keep_area & keep_tau

    # --- build filtered mask ---
    kept_ids = np.where(keep)[0] + 1  # label ids (1..N)
    final_mask = np.isin(labels_img, kept_ids).astype(np.uint8)

    # --- save ---
    save_mask_png(OUT_MASK_PNG, final_mask)

    print("✔ Done")
    print(f"  Structures found: {n}")
    print(f"  Kept (area >= {MIN_AREA_PX} and tau in ({low},{high}) ns): {int(keep.sum())}")
    print(f"  Saved filtered mask: {OUT_MASK_PNG}")
    print(f"  tau plane index used: {idx_tau}")


if __name__ == "__main__":
    main()