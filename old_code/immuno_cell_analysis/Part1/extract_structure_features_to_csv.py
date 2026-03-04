#!/usr/bin/env python3
"""
Extract per-structure morphology + lifetime/phasor metrics into a single CSV,
and save instance masks:
  - TIFF uint32 (exact IDs, analysis-ready)
  - PNG uint8 (visual inspection only)

Inputs:
- FINAL binary mask (structures > 0)
- phasor_data_CYX.tif (C,Y,X)

Outputs:
- CSV with per-structure features
- Instance mask TIFF (uint32)
- Instance mask PNG (uint8 visualization)

Run:
  python extract_structure_features_to_csv.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import tifffile as tiff
from PIL import Image
import csv


# =========================================================
# ===================== CONFIG ============================
# =========================================================
FINAL_MASK_PATH = Path(
    "/Users/schutyb/Documents/balu_lab/immuno_cell_analysis_data/visit_04/Mosaic07_4x4_FOV600_z150_32Sp/mask_filtered_area_tau_phase.png")
PHASOR_DATA_PATH = Path(
    "/Users/schutyb/Documents/balu_lab/immuno_cell_analysis_data/visit_04/Mosaic07_4x4_FOV600_z150_32Sp/phasor_data_CYX.tif")

OUT_CSV_PATH = Path(
    "/Users/schutyb/Documents/balu_lab/immuno_cell_analysis_data/visit_04/Mosaic07_4x4_FOV600_z150_32Sp/structure_features.csv")

OUT_INSTANCE_MASK_TIF = Path(
    "/Users/schutyb/Documents/balu_lab/immuno_cell_analysis_data/visit_04/Mosaic07_4x4_FOV600_z150_32Sp/structure_instance_mask_uint32.tif")
OUT_INSTANCE_MASK_PNG = Path(
    "/Users/schutyb/Documents/balu_lab/immuno_cell_analysis_data/visit_04/Mosaic07_4x4_FOV600_z150_32Sp/structure_instance_mask_preview.png")

CONNECTIVITY_8 = True
MIN_AREA_PX = 1
# =========================================================

# ---------------------------------------------------------
# IO
# ---------------------------------------------------------
def load_mask_2d(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext in [".tif", ".tiff"]:
        arr = tiff.imread(str(path))
    else:
        arr = np.array(Image.open(str(path)))

    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint8)


def read_phasor_data(path: Path) -> Tuple[np.ndarray, Optional[List[str]]]:
    arr = tiff.imread(str(path))
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
        raise ValueError(f"Expected (C,Y,X). Got shape={arr.shape}")

    return arr.astype(np.float32), labels


def plane_index(labels: Optional[List[str]], fallback: int, name: str) -> int:
    if isinstance(labels, list):
        try:
            return labels.index(name)
        except ValueError:
            return fallback
    return fallback


def save_instance_mask_tiff(path: Path, labels_img: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(str(path), labels_img.astype(np.uint32), dtype=np.uint32)


def save_instance_mask_png(path: Path, labels_img: np.ndarray):
    """
    Save instance labels as uint8 PNG for visualization only.
    IDs are rescaled to 1..255 if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    max_id = int(labels_img.max())
    if max_id == 0:
        png = np.zeros_like(labels_img, dtype=np.uint8)
    else:
        scale = 255.0 / max_id
        png = np.clip(labels_img * scale, 0, 255).astype(np.uint8)

    Image.fromarray(png, mode="L").save(str(path))


# ---------------------------------------------------------
# Connected components
# ---------------------------------------------------------
def connected_components(mask_bool: np.ndarray, connectivity_8: bool = True) -> Tuple[np.ndarray, List[int]]:
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


def relabel_keep_only(labels_img: np.ndarray, areas: np.ndarray, min_area: int):
    keep_old_ids = np.where(areas >= min_area)[0] + 1
    new_labels = np.zeros_like(labels_img, dtype=np.int32)

    for new_id, old_id in enumerate(keep_old_ids, start=1):
        new_labels[labels_img == old_id] = new_id

    return new_labels, keep_old_ids


# ---------------------------------------------------------
# Morphology helpers
# ---------------------------------------------------------
def perimeter_4n_pixel_edges(component_mask: np.ndarray) -> float:
    m = component_mask.astype(np.uint8)

    up = np.zeros_like(m);    up[1:, :] = m[:-1, :]
    dn = np.zeros_like(m);    dn[:-1, :] = m[1:, :]
    lf = np.zeros_like(m);    lf[:, 1:] = m[:, :-1]
    rt = np.zeros_like(m);    rt[:, :-1] = m[:, 1:]

    neighbors4 = up + dn + lf + rt
    return float(((4 - neighbors4) * m).sum())


def major_minor_axis_lengths(y: np.ndarray, x: np.ndarray):
    if y.size < 3:
        return 0.0, 0.0

    y = y.astype(float) - y.mean()
    x = x.astype(float) - x.mean()

    cov_xx = np.mean(x * x)
    cov_yy = np.mean(y * y)
    cov_xy = np.mean(x * y)

    trace = cov_xx + cov_yy
    det = cov_xx * cov_yy - cov_xy * cov_xy
    disc = max(trace * trace - 4 * det, 0.0)

    l1 = 0.5 * (trace + math.sqrt(disc))
    l2 = 0.5 * (trace - math.sqrt(disc))

    return 4 * math.sqrt(max(l1, 0)), 4 * math.sqrt(max(l2, 0))


def label_means(labels: np.ndarray, values: np.ndarray, n_labels: int):
    lab = labels.ravel()
    val = values.ravel().astype(float)

    fg = lab > 0
    sums = np.bincount(lab[fg], weights=val[fg], minlength=n_labels + 1)
    cnts = np.bincount(lab[fg], minlength=n_labels + 1)

    means = np.zeros(n_labels + 1)
    ok = cnts > 0
    means[ok] = sums[ok] / cnts[ok]
    return means[1:]


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    mask = load_mask_2d(FINAL_MASK_PATH) > 0
    labels_raw, areas_list = connected_components(mask, CONNECTIVITY_8)
    areas = np.array(areas_list, dtype=int)

    instance_labels, kept_old_ids = relabel_keep_only(labels_raw, areas, MIN_AREA_PX)
    K = int(instance_labels.max())
    if K == 0:
        raise ValueError("No structures left after filtering.")

    save_instance_mask_tiff(OUT_INSTANCE_MASK_TIF, instance_labels)
    save_instance_mask_png(OUT_INSTANCE_MASK_PNG, instance_labels)

    phasor, plane_labels = read_phasor_data(PHASOR_DATA_PATH)
    _, Y, X = phasor.shape
    if instance_labels.shape != (Y, X):
        raise ValueError("Mask and phasor_data shapes do not match.")

    idx = {
        "g_int": plane_index(plane_labels, 0, "green_mean"),
        "g_g":   plane_index(plane_labels, 1, "green_g"),
        "g_s":   plane_index(plane_labels, 2, "green_s"),
        "g_ph":  plane_index(plane_labels, 3, "green_phase"),
        "g_mod": plane_index(plane_labels, 4, "green_modulation"),
        "g_tp":  plane_index(plane_labels, 5, "green_tau_phase_ns"),
        "g_tm":  plane_index(plane_labels, 6, "green_tau_mod_ns"),
        "b_int": plane_index(plane_labels, 7, "blue_mean"),
        "b_g":   plane_index(plane_labels, 8, "blue_g"),
        "b_s":   plane_index(plane_labels, 9, "blue_s"),
        "b_ph":  plane_index(plane_labels,10, "blue_phase"),
        "b_mod": plane_index(plane_labels,11, "blue_modulation"),
        "b_tp":  plane_index(plane_labels,12, "blue_tau_phase_ns"),
        "b_tm":  plane_index(plane_labels,13, "blue_tau_mod_ns"),
    }

    # Precompute means
    gm = {k: label_means(instance_labels, phasor[i], K) for k, i in idx.items()}

    OUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "structure_id",
            "area_px", "equivalent_diameter_px", "perimeter_px",
            "major_axis_px", "minor_axis_px", "axis_ratio", "circularity",
            "green_intensity_mean", "green_g_mean", "green_s_mean",
            "green_phase_mean", "green_modulation_mean",
            "green_tau_phase_mean_ns", "green_tau_mod_mean_ns",
            "blue_intensity_mean", "blue_g_mean", "blue_s_mean",
            "blue_phase_mean", "blue_modulation_mean",
            "blue_tau_phase_mean_ns", "blue_tau_mod_mean_ns",
        ])

        for sid in range(1, K + 1):
            comp = instance_labels == sid
            area = int(comp.sum())

            eq_d = math.sqrt(4 * area / math.pi)
            per = perimeter_4n_pixel_edges(comp)
            circ = 4 * math.pi * area / (per * per) if per > 0 else 0.0

            y, x = np.nonzero(comp)
            maj, minr = major_minor_axis_lengths(y, x)
            ratio = maj / minr if minr > 0 else float("inf")

            w.writerow([
                sid,
                area, eq_d, per, maj, minr, ratio, circ,
                gm["g_int"][sid-1], gm["g_g"][sid-1], gm["g_s"][sid-1],
                gm["g_ph"][sid-1], gm["g_mod"][sid-1],
                gm["g_tp"][sid-1], gm["g_tm"][sid-1],
                gm["b_int"][sid-1], gm["b_g"][sid-1], gm["b_s"][sid-1],
                gm["b_ph"][sid-1], gm["b_mod"][sid-1],
                gm["b_tp"][sid-1], gm["b_tm"][sid-1],
            ])

    print("✔ Done")
    print(f"  Structures exported: {K}")
    print(f"  CSV: {OUT_CSV_PATH}")
    print(f"  Instance mask TIFF: {OUT_INSTANCE_MASK_TIF}")
    print(f"  Instance mask PNG:  {OUT_INSTANCE_MASK_PNG}")


if __name__ == "__main__":
    main()