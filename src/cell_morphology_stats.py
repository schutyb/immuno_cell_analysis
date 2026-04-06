#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import imageio.v3 as iio
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt

from skimage.measure import label, regionprops
from scipy.stats import mannwhitneyu


# ============================================================
# CONFIG
# ============================================================

PATIENT_DIR = Path("/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449")
ANALYSIS_DIR = PATIENT_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

ROI_LABELS_CSV = ANALYSIS_DIR / "roi_phasor_points_with_gmm_labels_all_three_types.csv"

PHASOR_TYPE_TO_USE = "coumarin_calibrated"
BIO_LABEL_TO_KEEP = "cells"

PIXEL_SIZE_UM = 0.5

OUTPUT_DIR = ANALYSIS_DIR / "cell_morphology_stats"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PER_CELL_CSV = OUTPUT_DIR / "cell_morphology_per_cell.csv"
PER_VISIT_SUMMARY_CSV = OUTPUT_DIR / "cell_morphology_summary_by_visit.csv"
STATS_CSV = OUTPUT_DIR / "cell_morphology_stats_vs_visit01.csv"
TOP_CHANGES_CSV = OUTPUT_DIR / "cell_morphology_top_changes.csv"

SHOW_PLOTS = False


# ============================================================
# HELPERS
# ============================================================

def find_instance_mask(mosaic_dir: Path) -> Optional[Path]:
    candidates = [
        mosaic_dir / "_new" / "cells_instance_mask.tif",
        mosaic_dir / "cells_instance_mask.tif",
        mosaic_dir / "_new" / "instance_mask_filtered.tif",
        mosaic_dir / "_new" / "instance_mask_filtered.tiff",
        mosaic_dir / "_new" / "instance_mask_filtered.png",
        mosaic_dir / "instance_mask_filtered.tif",
        mosaic_dir / "instance_mask_filtered.tiff",
        mosaic_dir / "instance_mask_filtered.png",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def read_mask(mask_path: Path) -> np.ndarray:
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
        return label(mask > 0).astype(np.int32)

    return mask.astype(np.int32)


def infer_visit_from_path(path: Path) -> str:
    m = re.search(r"(visit[_-]?\d+)", str(path), re.IGNORECASE)
    return m.group(1).lower() if m else "unknown_visit"


def build_cells_instance_mask_from_labels(
    instance_mask: np.ndarray,
    df_labels_subset: pd.DataFrame,
) -> np.ndarray:
    keep_labels = np.sort(df_labels_subset["roi_label"].astype(int).unique())
    return np.where(np.isin(instance_mask, keep_labels), instance_mask, 0).astype(np.int32)


def relabel_sequential(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=np.int32)
    labels = np.unique(mask)
    labels = labels[labels > 0]
    for new_id, old_id in enumerate(labels, start=1):
        out[mask == old_id] = new_id
    return out


def compute_morphology_from_mask(
    mask: np.ndarray,
    visit: str,
    mosaic_name: str,
    source_mask_path: str,
) -> pd.DataFrame:
    props = regionprops(mask)

    rows = []
    for prop in props:
        area_px = float(prop.area)
        perimeter_px = float(prop.perimeter) if prop.perimeter is not None else np.nan
        major_px = float(prop.major_axis_length) if prop.major_axis_length is not None else np.nan
        minor_px = float(prop.minor_axis_length) if prop.minor_axis_length is not None else np.nan
        eq_diameter_px = float(prop.equivalent_diameter_area)
        eccentricity = float(prop.eccentricity) if prop.eccentricity is not None else np.nan
        solidity = float(prop.solidity) if prop.solidity is not None else np.nan

        area_um2 = area_px * (PIXEL_SIZE_UM ** 2)
        perimeter_um = perimeter_px * PIXEL_SIZE_UM if np.isfinite(perimeter_px) else np.nan
        major_um = major_px * PIXEL_SIZE_UM if np.isfinite(major_px) else np.nan
        minor_um = minor_px * PIXEL_SIZE_UM if np.isfinite(minor_px) else np.nan
        eq_diameter_um = eq_diameter_px * PIXEL_SIZE_UM

        circularity = np.nan
        if np.isfinite(perimeter_px) and perimeter_px > 0:
            circularity = 4.0 * np.pi * area_px / (perimeter_px ** 2)

        aspect_ratio = np.nan
        if np.isfinite(major_px) and np.isfinite(minor_px) and minor_px > 0:
            aspect_ratio = major_px / minor_px

        rows.append({
            "visit": visit,
            "mosaic_name": mosaic_name,
            "cell_label": int(prop.label),
            "source_mask_path": source_mask_path,
            "area_um2": area_um2,
            "equivalent_diameter_um": eq_diameter_um,
            "perimeter_um": perimeter_um,
            "major_axis_length_um": major_um,
            "minor_axis_length_um": minor_um,
            "circularity": circularity,
            "eccentricity": eccentricity,
            "aspect_ratio": aspect_ratio,
            "solidity": solidity,
        })

    return pd.DataFrame(rows)


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    sx = np.var(x, ddof=1)
    sy = np.var(y, ddof=1)
    pooled = np.sqrt(((len(x)-1)*sx + (len(y)-1)*sy) / (len(x)+len(y)-2))
    if pooled == 0:
        return np.nan
    return (np.mean(y) - np.mean(x)) / pooled


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = np.empty(n, dtype=float)

    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adj[i] = prev

    out = np.empty(n, dtype=float)
    out[order] = np.clip(adj, 0, 1)
    return out


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print(f"[INFO] Patient dir: {PATIENT_DIR}")
    print(f"[INFO] Analysis dir: {ANALYSIS_DIR}")
    print(f"[INFO] ROI labels CSV: {ROI_LABELS_CSV}")

    if not ROI_LABELS_CSV.exists():
        raise FileNotFoundError(f"Missing CSV: {ROI_LABELS_CSV}")

    df_labels = pd.read_csv(ROI_LABELS_CSV)
    print(f"[INFO] Loaded ROI labels rows: {len(df_labels)}")
    print(f"[INFO] Columns: {list(df_labels.columns)}")

    required_cols = {"visit", "mosaic_name", "roi_label", "bio_label"}
    missing = required_cols - set(df_labels.columns)
    if missing:
        raise ValueError(f"Missing columns in ROI labels CSV: {sorted(missing)}")

    if "phasor_type" in df_labels.columns:
        before = len(df_labels)
        df_labels = df_labels[df_labels["phasor_type"] == PHASOR_TYPE_TO_USE].copy()
        print(f"[INFO] Filtered by phasor_type='{PHASOR_TYPE_TO_USE}': {before} -> {len(df_labels)}")

    df_labels["visit"] = df_labels["visit"].astype(str).str.lower()
    df_labels["mosaic_name"] = df_labels["mosaic_name"].astype(str)
    df_labels["bio_label"] = df_labels["bio_label"].astype(str).str.lower()

    print("[INFO] Visits in labels CSV:", sorted(df_labels["visit"].unique()))
    print("[INFO] Mosaic names example:", df_labels["mosaic_name"].head().tolist())

    all_cells = []
    total_mosaics_seen = 0
    total_masks_found = 0
    total_mosaics_used = 0

    visit_dirs = sorted([p for p in PATIENT_DIR.iterdir() if p.is_dir() and p.name.lower().startswith("visit")])
    print(f"[INFO] Visit dirs found: {len(visit_dirs)}")

    for visit_dir in visit_dirs:
        print(f"\n[INFO] Processing {visit_dir.name}")
        mosaic_dirs = sorted([p for p in visit_dir.iterdir() if p.is_dir() and p.name.lower().startswith("mosaic")])
        print(f"[INFO] Mosaic dirs found in {visit_dir.name}: {len(mosaic_dirs)}")

        for mosaic_dir in mosaic_dirs:
            total_mosaics_seen += 1
            visit = infer_visit_from_path(mosaic_dir)
            mosaic_name = mosaic_dir.name

            print(f"  [CHECK] {visit} | {mosaic_name}")

            mask_path = find_instance_mask(mosaic_dir)
            if mask_path is None:
                print("    -> no mask found")
                continue

            total_masks_found += 1
            print(f"    -> mask: {mask_path.name}")

            mask = read_mask(mask_path)

            if mask_path.name == "cells_instance_mask.tif":
                print("    -> using precomputed cells_instance_mask.tif")
                cells_mask = relabel_sequential(mask)
            else:
                df_subset = df_labels[
                    (df_labels["visit"] == visit) &
                    (df_labels["mosaic_name"] == mosaic_name) &
                    (df_labels["bio_label"] == BIO_LABEL_TO_KEEP)
                ].copy()

                print(f"    -> matching ROI labels for cells: {len(df_subset)}")

                if df_subset.empty:
                    print("    -> no matching cell labels in CSV")
                    continue

                cells_mask = build_cells_instance_mask_from_labels(mask, df_subset)
                cells_mask = relabel_sequential(cells_mask)

            n_cells_mask = int(np.max(cells_mask))
            print(f"    -> segmented cells in mask: {n_cells_mask}")

            if n_cells_mask == 0:
                print("    -> empty cells mask after filtering")
                continue

            df_cells = compute_morphology_from_mask(
                cells_mask,
                visit=visit,
                mosaic_name=mosaic_name,
                source_mask_path=str(mask_path),
            )

            print(f"    -> morphology rows: {len(df_cells)}")

            if len(df_cells) > 0:
                all_cells.append(df_cells)
                total_mosaics_used += 1

    print("\n[SUMMARY]")
    print(f"  Total mosaics seen:  {total_mosaics_seen}")
    print(f"  Total masks found:   {total_masks_found}")
    print(f"  Total mosaics used:  {total_mosaics_used}")

    if not all_cells:
        raise RuntimeError("No cell morphology data could be generated.")

    df_all = pd.concat(all_cells, ignore_index=True)
    print(f"[INFO] Total cells in final table: {len(df_all)}")
    df_all.to_csv(PER_CELL_CSV, index=False)

    features = [
        "area_um2",
        "equivalent_diameter_um",
        "circularity",
        "eccentricity",
        "aspect_ratio",
        "solidity",
    ]

    # --------------------------------------------------------
    # summary by visit
    # --------------------------------------------------------
    summary_rows = []
    for visit in sorted(df_all["visit"].unique()):
        dv = df_all[df_all["visit"] == visit].copy()

        row = {
            "visit": visit,
            "n_cells": len(dv),
            "n_mosaics": dv["mosaic_name"].nunique(),
        }

        for feat in features:
            vals = dv[feat].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                row[f"{feat}_mean"] = np.nan
                row[f"{feat}_median"] = np.nan
                row[f"{feat}_std"] = np.nan
                row[f"{feat}_q25"] = np.nan
                row[f"{feat}_q75"] = np.nan
            else:
                row[f"{feat}_mean"] = float(np.mean(vals))
                row[f"{feat}_median"] = float(np.median(vals))
                row[f"{feat}_std"] = float(np.std(vals, ddof=0))
                row[f"{feat}_q25"] = float(np.percentile(vals, 25))
                row[f"{feat}_q75"] = float(np.percentile(vals, 75))

        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    print(f"[INFO] Summary rows: {len(df_summary)}")
    df_summary.to_csv(PER_VISIT_SUMMARY_CSV, index=False)

    # --------------------------------------------------------
    # stats vs visit01
    # --------------------------------------------------------
    ref_visit = "visit01"
    if ref_visit not in df_all["visit"].unique():
        raise RuntimeError("visit01 not found in morphology table.")

    dref = df_all[df_all["visit"] == ref_visit].copy()

    stat_rows = []
    for visit in sorted(df_all["visit"].unique()):
        if visit == ref_visit:
            continue

        dv = df_all[df_all["visit"] == visit].copy()

        for feat in features:
            x = dref[feat].to_numpy(dtype=float)
            y = dv[feat].to_numpy(dtype=float)

            x = x[np.isfinite(x)]
            y = y[np.isfinite(y)]

            if len(x) < 3 or len(y) < 3:
                pval = np.nan
            else:
                try:
                    _, pval = mannwhitneyu(x, y, alternative="two-sided")
                except Exception:
                    pval = np.nan

            stat_rows.append({
                "reference_visit": ref_visit,
                "comparison_visit": visit,
                "feature": feat,
                "ref_median": float(np.nanmedian(x)) if len(x) else np.nan,
                "cmp_median": float(np.nanmedian(y)) if len(y) else np.nan,
                "median_diff": float(np.nanmedian(y) - np.nanmedian(x)) if len(x) and len(y) else np.nan,
                "ref_mean": float(np.nanmean(x)) if len(x) else np.nan,
                "cmp_mean": float(np.nanmean(y)) if len(y) else np.nan,
                "cohen_d": cohen_d(x, y),
                "p_value": pval,
                "n_ref": len(x),
                "n_cmp": len(y),
            })

    df_stats = pd.DataFrame(stat_rows)

    if len(df_stats) > 0:
        df_stats["p_adj_fdr"] = np.nan
        for visit in sorted(df_stats["comparison_visit"].unique()):
            mask = df_stats["comparison_visit"] == visit
            pvals = df_stats.loc[mask, "p_value"].to_numpy(dtype=float)
            valid = np.isfinite(pvals)
            adj = np.full_like(pvals, np.nan, dtype=float)
            if np.any(valid):
                adj_valid = bh_fdr(pvals[valid])
                adj[valid] = adj_valid
            df_stats.loc[mask, "p_adj_fdr"] = adj

        df_stats["significant_fdr_0_05"] = df_stats["p_adj_fdr"] < 0.05
        df_stats.to_csv(STATS_CSV, index=False)

        df_top = df_stats.copy()
        df_top["abs_cohen_d"] = np.abs(df_top["cohen_d"])
        df_top = df_top.sort_values(
            ["comparison_visit", "significant_fdr_0_05", "abs_cohen_d"],
            ascending=[True, False, False]
        )
        df_top.to_csv(TOP_CHANGES_CSV, index=False)

    print(f"[DONE] Saved per-cell table: {PER_CELL_CSV}")
    print(f"[DONE] Saved summary table: {PER_VISIT_SUMMARY_CSV}")
    if len(df_stats) > 0:
        print(f"[DONE] Saved stats table: {STATS_CSV}")
        print(f"[DONE] Saved ranked changes: {TOP_CHANGES_CSV}")
    print(f"[DONE] Output dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()