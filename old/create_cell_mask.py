#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import tifffile
import imageio.v3 as iio
from skimage.measure import label


# =========================
# CONFIG
# =========================

ROOT_DIR = Path("/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449")
ANALYSIS_DIR = ROOT_DIR / "analysis"

ROI_LABELS_CSV = ANALYSIS_DIR / "roi_phasor_points_with_gmm_labels_corrected.csv"

SAVE_BINARY_PNG = True
SAVE_BINARY_TIF = True
SAVE_INSTANCE_TIF = True


# =========================
# HELPERS
# =========================

def infer_patient_visit_from_path(path: Path) -> Tuple[str, str]:
    path_str = str(path)

    patient_match = re.search(r"(p\d+)", path_str, re.IGNORECASE)
    visit_match = re.search(r"(visit[_-]?\d+)", path_str, re.IGNORECASE)

    patient = patient_match.group(1) if patient_match else "unknown_patient"
    visit = visit_match.group(1) if visit_match else "unknown_visit"
    return patient, visit


def find_matching_mask(folder: Path) -> Optional[Path]:
    candidates = [
        folder / "instance_mask_filtered.tif",
        folder / "instance_mask_filtered.tiff",
        folder / "instance_mask_filtered.png",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def read_mask(mask_path: Path) -> np.ndarray:
    """
    Read instance mask from tif/png.
    If binary, connected components are labeled.
    If already labeled, use labels directly.
    """
    if mask_path.suffix.lower() in {".tif", ".tiff"}:
        mask = tifffile.imread(mask_path)
    else:
        mask = iio.imread(mask_path)

    mask = np.asarray(mask).squeeze()

    if mask.ndim == 3:
        if mask.shape[-1] in (3, 4):
            if np.all(mask[..., 0] == mask[..., 1]):
                mask = mask[..., 0]
            else:
                mask = mask[..., :3].max(axis=-1)
        else:
            raise ValueError(f"Unsupported mask shape: {mask.shape} in {mask_path}")

    if mask.dtype == bool:
        return label(mask).astype(np.int32)

    unique_vals = np.unique(mask)
    if set(unique_vals.tolist()).issubset({0, 1, 255}):
        binary = mask > 0
        return label(binary).astype(np.int32)

    return mask.astype(np.int32)


def make_cells_instance_mask(instance_mask: np.ndarray, cell_labels: np.ndarray) -> np.ndarray:
    """
    Keep only ROI labels belonging to cells, preserving instance labels.
    """
    cell_labels = np.asarray(cell_labels, dtype=np.int32)
    keep = np.isin(instance_mask, cell_labels)
    return np.where(keep, instance_mask, 0).astype(np.int32)


def relabel_sequential(mask: np.ndarray) -> np.ndarray:
    """
    Relabel non-zero instance labels to 1..N sequentially.
    """
    out = np.zeros_like(mask, dtype=np.int32)
    labels = np.unique(mask)
    labels = labels[labels > 0]

    for new_id, old_id in enumerate(labels, start=1):
        out[mask == old_id] = new_id

    return out


# =========================
# MAIN
# =========================

def main() -> None:
    if not ROI_LABELS_CSV.exists():
        raise FileNotFoundError(f"Missing CSV: {ROI_LABELS_CSV}")

    df = pd.read_csv(ROI_LABELS_CSV)

    required_cols = {"visit", "roi_label", "bio_label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    visit_dirs = sorted(
        [p for p in ROOT_DIR.iterdir() if p.is_dir() and p.name.lower().startswith("visit")]
    )

    if not visit_dirs:
        raise RuntimeError(f"No visit folders found in: {ROOT_DIR}")

    for visit_dir in visit_dirs:
        visit_name = visit_dir.name

        mask_path = find_matching_mask(visit_dir)
        if mask_path is None:
            print(f"[SKIP] {visit_name}: no instance_mask_filtered found")
            continue

        df_visit = df[df["visit"].str.lower() == visit_name.lower()].copy()
        if df_visit.empty:
            print(f"[SKIP] {visit_name}: no rows in CSV")
            continue

        df_cells = df_visit[df_visit["bio_label"] == "cells"].copy()
        if df_cells.empty:
            print(f"[SKIP] {visit_name}: no ROIs labeled as cells")
            continue

        instance_mask = read_mask(mask_path)

        cell_roi_labels = np.sort(df_cells["roi_label"].astype(int).unique())
        cells_instance_mask = make_cells_instance_mask(instance_mask, cell_roi_labels)

        # optional: relabel to 1..N for easier downstream processing
        cells_instance_mask_seq = relabel_sequential(cells_instance_mask)

        cells_binary_mask = (cells_instance_mask_seq > 0).astype(np.uint8)

        n_cells = len(np.unique(cells_instance_mask_seq)) - 1
        print(f"[OK] {visit_name}: {n_cells} cell ROIs")

        if SAVE_BINARY_PNG:
            out_png = visit_dir / "cells_mask.png"
            iio.imwrite(out_png, (cells_binary_mask * 255).astype(np.uint8))

        if SAVE_BINARY_TIF:
            out_tif = visit_dir / "cells_mask.tif"
            tifffile.imwrite(out_tif, cells_binary_mask.astype(np.uint8))

        if SAVE_INSTANCE_TIF:
            out_inst = visit_dir / "cells_instance_mask.tif"
            tifffile.imwrite(out_inst, cells_instance_mask_seq.astype(np.int32))

    print("[DONE]")


if __name__ == "__main__":
    main()