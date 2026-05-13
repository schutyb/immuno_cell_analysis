#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Global GMM segmentation of area-filtered binary ROI masks
using corrected green-channel phasor.

Input per mosaic:
    phasor/phasor_raw_green_blue_mosaic_elastin_corrected.tif
    segmentation_area_phasor/*_mask_area_filtered.tif

Important:
    *_mask_area_filtered.tif is binary.
    Therefore, connected components are labeled first to recover individual ROIs.

Output:
    analysis/global_gmm_green_corrected_from_area_masks/
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

from skimage.measure import label, regionprops
from sklearn.mixture import GaussianMixture


# ============================================================
# CONFIG
# ============================================================

ROOT_DIR = Path(
    "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449"
).expanduser()

OUTPUT_DIR = ROOT_DIR / "analysis" / "global_gmm_green_corrected_from_area_masks_relabel"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PHASOR_FILE_PATTERNS = [
    "phasor/phasor_raw_green_blue_mosaic_elastin_corrected.tif",
]

MASK_FILE_PATTERNS = [
    "segmentation_area_phasor/*_mask_area_filtered.tif",
]

# Corrected GREEN phasor channels.
# Expected stack:
# 0 = intensity/DC green
# 1 = G green corrected
# 2 = S green corrected
# 3 = intensity/DC blue
# 4 = G blue corrected
# 5 = S blue corrected
G_INDEX = 1
S_INDEX = 2

MIN_AREA = 50
MAX_AREA = None

N_COMPONENTS = 3
RANDOM_STATE = 42

CLASS_COLORS = {
    "melanin": "tab:blue",
    "cells": "tab:red",
    "elastin": "tab:green",
}

CLASS_ID = {
    "melanin": 1,
    "cells": 2,
    "elastin": 3,
}

PHASOR_XLIM = None
PHASOR_YLIM = None


# ============================================================
# HELPERS
# ============================================================

def find_first_matching(folder: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(folder.glob(pattern))
        if matches:
            return matches[0]

        matches = sorted(folder.rglob(pattern))
        if matches:
            return matches[0]

    return None


def parse_visit(path: Path) -> str:
    match = re.search(r"visit\d+", str(path))
    return match.group(0) if match else "unknown_visit"


def load_green_phasor(path: Path):
    arr = tiff.imread(path)

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D phasor stack, got shape {arr.shape}")

    if arr.shape[0] <= 10:
        g = arr[G_INDEX].astype(np.float32)
        s = arr[S_INDEX].astype(np.float32)
    elif arr.shape[-1] <= 10:
        g = arr[..., G_INDEX].astype(np.float32)
        s = arr[..., S_INDEX].astype(np.float32)
    else:
        raise ValueError(f"Cannot infer channel axis from shape {arr.shape}")

    return g, s


def get_mosaic_dirs(root_dir: Path) -> list[Path]:
    return sorted(
        p for p in root_dir.glob("visit*/Mosaic*")
        if p.is_dir()
    )


def load_binary_mask_as_instances(mask_path: Path) -> np.ndarray:
    mask = tiff.imread(mask_path)

    if mask.ndim > 2:
        mask = np.squeeze(mask)

    binary_mask = mask > 0

    labeled_mask = label(binary_mask, connectivity=2).astype(np.int32)

    return labeled_mask


def build_roi_table_for_mosaic(mosaic_dir: Path) -> pd.DataFrame:
    visit = parse_visit(mosaic_dir)
    mosaic_name = mosaic_dir.name

    phasor_path = find_first_matching(mosaic_dir, PHASOR_FILE_PATTERNS)
    mask_path = find_first_matching(mosaic_dir, MASK_FILE_PATTERNS)

    if phasor_path is None:
        print(f"[SKIP] No corrected phasor found: {visit} | {mosaic_name}")
        return pd.DataFrame()

    if mask_path is None:
        print(f"[SKIP] No area-filtered mask found: {visit} | {mosaic_name}")
        return pd.DataFrame()

    g_img, s_img = load_green_phasor(phasor_path)
    labeled_mask = load_binary_mask_as_instances(mask_path)

    if labeled_mask.shape != g_img.shape:
        raise ValueError(
            f"Shape mismatch in {visit} | {mosaic_name}\n"
            f"mask:   {labeled_mask.shape}\n"
            f"phasor: {g_img.shape}\n"
            f"mask path: {mask_path}\n"
            f"phasor path: {phasor_path}"
        )

    rows = []

    for prop in regionprops(labeled_mask):
        roi_label = int(prop.label)
        area = int(prop.area)

        if area < MIN_AREA:
            continue

        if MAX_AREA is not None and area > MAX_AREA:
            continue

        coords = prop.coords
        yy = coords[:, 0]
        xx = coords[:, 1]

        g_vals = g_img[yy, xx]
        s_vals = s_img[yy, xx]

        valid = np.isfinite(g_vals) & np.isfinite(s_vals)

        if valid.sum() == 0:
            continue

        g_mean = float(np.mean(g_vals[valid]))
        s_mean = float(np.mean(s_vals[valid]))
        phase = float(np.arctan2(s_mean, g_mean))

        rows.append(
            {
                "visit": visit,
                "mosaic": mosaic_name,
                "mosaic_dir": str(mosaic_dir),
                "phasor_path": str(phasor_path),
                "mask_path": str(mask_path),
                "roi_label": roi_label,
                "area_px": area,
                "centroid_y": float(prop.centroid[0]),
                "centroid_x": float(prop.centroid[1]),
                "g_green_corrected": g_mean,
                "s_green_corrected": s_mean,
                "phase_rad": phase,
            }
        )

    print(f"[OK] {visit} | {mosaic_name} | ROIs kept: {len(rows)}")

    return pd.DataFrame(rows)


def assign_cluster_names(df: pd.DataFrame) -> dict[int, str]:
    phase_by_cluster = (
        df.groupby("gmm_cluster")["phase_rad"]
        .median()
        .sort_values()
    )

    ordered_clusters = list(phase_by_cluster.index)

    return {
        ordered_clusters[0]: "melanin",
        ordered_clusters[1]: "cells",
        ordered_clusters[2]: "elastin",
    }


def plot_phasor(df: pd.DataFrame, out_path: Path, title: str):
    plt.figure(figsize=(7, 6))

    for class_name, color in CLASS_COLORS.items():
        sub = df[df["class_name"] == class_name]

        if sub.empty:
            continue

        plt.scatter(
            sub["g_green_corrected"],
            sub["s_green_corrected"],
            s=6,
            alpha=0.45,
            c=color,
            label=f"{class_name} (n={len(sub)})",
            edgecolors="none",
        )

    theta = np.linspace(0, np.pi, 400)
    x = 0.5 + 0.5 * np.cos(theta)
    y = 0.5 * np.sin(theta)
    plt.plot(x, y, "k--", linewidth=1)

    plt.xlabel("G green corrected")
    plt.ylabel("S green corrected")
    plt.title(title)
    plt.legend(frameon=False)
    plt.gca().set_aspect("equal", adjustable="box")

    if PHASOR_XLIM is not None:
        plt.xlim(*PHASOR_XLIM)

    if PHASOR_YLIM is not None:
        plt.ylim(*PHASOR_YLIM)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_masks_for_mosaic(mosaic_df: pd.DataFrame):
    if mosaic_df.empty:
        return

    visit = mosaic_df["visit"].iloc[0]
    mosaic_name = mosaic_df["mosaic"].iloc[0]
    mask_path = Path(mosaic_df["mask_path"].iloc[0])

    labeled_mask = load_binary_mask_as_instances(mask_path)

    out_dir = OUTPUT_DIR / visit / mosaic_name
    out_dir.mkdir(parents=True, exist_ok=True)

    multiclass_mask = np.zeros(labeled_mask.shape, dtype=np.uint8)

    for class_name, class_value in CLASS_ID.items():
        roi_labels = mosaic_df.loc[
            mosaic_df["class_name"] == class_name,
            "roi_label",
        ].astype(int).to_numpy()

        binary_mask = np.isin(labeled_mask, roi_labels).astype(np.uint8)

        tiff.imwrite(
            out_dir / f"{class_name}_mask.tif",
            binary_mask,
        )

        multiclass_mask[binary_mask > 0] = class_value

    tiff.imwrite(
        out_dir / "multiclass_mask_melanin1_cells2_elastin3.tif",
        multiclass_mask,
    )

    plot_phasor(
        mosaic_df,
        out_dir / "phasor_segmented_green_corrected.png",
        title=f"{visit} | {mosaic_name} | green corrected GMM",
    )

    mosaic_df.to_csv(
        out_dir / "roi_phasor_gmm_green_corrected.csv",
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    mosaic_dirs = get_mosaic_dirs(ROOT_DIR)

    print(f"\nFound {len(mosaic_dirs)} mosaic folders.\n")

    all_tables = []

    for mosaic_dir in mosaic_dirs:
        df = build_roi_table_for_mosaic(mosaic_dir)

        if not df.empty:
            all_tables.append(df)

    if len(all_tables) == 0:
        raise RuntimeError("No valid ROIs found.")

    df_all = pd.concat(all_tables, ignore_index=True)

    print(f"\nTotal ROIs for global GMM: {len(df_all)}")

    X = df_all[
        ["g_green_corrected", "s_green_corrected"]
    ].to_numpy(dtype=np.float32)

    gmm = GaussianMixture(
        n_components=N_COMPONENTS,
        covariance_type="full",
        random_state=RANDOM_STATE,
    )

    df_all["gmm_cluster"] = gmm.fit_predict(X)
    probs = gmm.predict_proba(X)
    df_all["gmm_probability"] = probs.max(axis=1)

    cluster_to_name = assign_cluster_names(df_all)
    df_all["class_name"] = df_all["gmm_cluster"].map(cluster_to_name)

    print("\nGlobal cluster assignment:")
    for cluster_id, class_name in cluster_to_name.items():
        sub = df_all[df_all["gmm_cluster"] == cluster_id]
        print(
            f"  Cluster {cluster_id} -> {class_name} | "
            f"n={len(sub)} | "
            f"median phase={sub['phase_rad'].median():.4f} | "
            f"median G={sub['g_green_corrected'].median():.4f} | "
            f"median S={sub['s_green_corrected'].median():.4f}"
        )

    df_all.to_csv(
        OUTPUT_DIR / "global_roi_phasor_gmm_green_corrected.csv",
        index=False,
    )

    plot_phasor(
        df_all,
        OUTPUT_DIR / "global_phasor_segmented_green_corrected.png",
        title="Global GMM segmentation | green corrected phasor | all visits",
    )

    summary = (
        df_all.groupby(["visit", "mosaic", "class_name"])
        .size()
        .reset_index(name="n_rois")
    )

    summary.to_csv(
        OUTPUT_DIR / "global_gmm_class_counts_by_mosaic.csv",
        index=False,
    )

    for _, mosaic_df in df_all.groupby(["visit", "mosaic"], sort=True):
        save_masks_for_mosaic(mosaic_df)

    print("\nDone.")
    print(f"Results saved in:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()