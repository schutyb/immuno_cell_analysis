#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, List

import imageio.v3 as iio
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt

from skimage.measure import label, regionprops, regionprops_table


# ============================================================
# CONFIG
# ============================================================

PATIENT_DIR = Path("/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449")
ANALYSIS_DIR = PATIENT_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# CSV with ROI-level GMM labels
ROI_LABELS_CSV = ANALYSIS_DIR / "roi_phasor_points_with_gmm_labels_all_three_types.csv"

# Which phasor pipeline to use for defining "cells"
PHASOR_TYPE_TO_USE = "coumarin_calibrated"
BIO_LABEL_TO_KEEP = "cells"

PIXEL_SIZE_UM = 0.5

OUTPUT_DIR = ANALYSIS_DIR / "cell_morphology_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PER_CELL_CSV = OUTPUT_DIR / "cell_morphology_per_cell.csv"
PER_VISIT_SUMMARY_CSV = OUTPUT_DIR / "cell_morphology_summary_by_visit.csv"
SHIFT_TABLE_CSV = OUTPUT_DIR / "cell_morphology_shift_vs_visit01.csv"

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


def infer_mosaic_name_from_path(path: Path) -> str:
    for part in path.parts:
        if part.lower().startswith("mosaic"):
            return part
    return path.parent.name


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
            "area_px": area_px,
            "area_um2": area_um2,
            "perimeter_px": perimeter_px,
            "perimeter_um": perimeter_um,
            "equivalent_diameter_um": eq_diameter_um,
            "major_axis_length_um": major_um,
            "minor_axis_length_um": minor_um,
            "circularity": circularity,
            "eccentricity": eccentricity,
            "aspect_ratio": aspect_ratio,
            "solidity": solidity,
            "centroid_row": float(prop.centroid[0]),
            "centroid_col": float(prop.centroid[1]),
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


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not ROI_LABELS_CSV.exists():
        raise FileNotFoundError(f"Missing CSV: {ROI_LABELS_CSV}")

    df_labels = pd.read_csv(ROI_LABELS_CSV)

    required_cols = {"visit", "mosaic_name", "roi_label", "bio_label"}
    missing = required_cols - set(df_labels.columns)
    if missing:
        raise ValueError(f"Missing columns in ROI labels CSV: {sorted(missing)}")

    if "phasor_type" in df_labels.columns:
        df_labels = df_labels[df_labels["phasor_type"] == PHASOR_TYPE_TO_USE].copy()

    df_labels["visit"] = df_labels["visit"].astype(str).str.lower()
    df_labels["mosaic_name"] = df_labels["mosaic_name"].astype(str)

    all_cells = []

    for mosaic_dir in sorted(PATIENT_DIR.rglob("Mosaic*")):
        if not mosaic_dir.is_dir():
            continue

        visit = infer_visit_from_path(mosaic_dir)
        mosaic_name = mosaic_dir.name

        mask_path = find_instance_mask(mosaic_dir)
        if mask_path is None:
            continue

        mask = read_mask(mask_path)

        # if already a cell-only instance mask, use it directly
        if mask_path.name == "cells_instance_mask.tif":
            cells_mask = relabel_sequential(mask)
        else:
            df_subset = df_labels[
                (df_labels["visit"] == visit) &
                (df_labels["mosaic_name"] == mosaic_name) &
                (df_labels["bio_label"].str.lower() == BIO_LABEL_TO_KEEP)
            ].copy()

            if df_subset.empty:
                continue

            cells_mask = build_cells_instance_mask_from_labels(mask, df_subset)
            cells_mask = relabel_sequential(cells_mask)

        if np.max(cells_mask) == 0:
            continue

        df_cells = compute_morphology_from_mask(
            cells_mask,
            visit=visit,
            mosaic_name=mosaic_name,
            source_mask_path=str(mask_path),
        )

        if len(df_cells) > 0:
            all_cells.append(df_cells)

    if not all_cells:
        raise RuntimeError("No cell morphology data could be generated.")

    df_all = pd.concat(all_cells, ignore_index=True)
    df_all.to_csv(PER_CELL_CSV, index=False)

    # --------------------------------------------------------
    # summary by visit
    # --------------------------------------------------------
    features = [
        "area_um2",
        "equivalent_diameter_um",
        "circularity",
        "eccentricity",
        "aspect_ratio",
        "solidity",
    ]

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
    df_summary.to_csv(PER_VISIT_SUMMARY_CSV, index=False)

    # --------------------------------------------------------
    # shift vs visit01
    # --------------------------------------------------------
    shift_rows = []
    ref_visit = "visit01"
    if ref_visit in df_all["visit"].unique():
        dref = df_all[df_all["visit"] == ref_visit].copy()
        for visit in sorted(df_all["visit"].unique()):
            if visit == ref_visit:
                continue
            dv = df_all[df_all["visit"] == visit].copy()

            row = {"reference_visit": ref_visit, "comparison_visit": visit}
            for feat in features:
                row[f"{feat}_cohen_d"] = cohen_d(dref[feat].values, dv[feat].values)
                row[f"{feat}_median_diff"] = float(np.nanmedian(dv[feat]) - np.nanmedian(dref[feat]))
            shift_rows.append(row)

    df_shift = pd.DataFrame(shift_rows)
    df_shift.to_csv(SHIFT_TABLE_CSV, index=False)

    # --------------------------------------------------------
    # plots
    # --------------------------------------------------------
    ordered_visits = [v for v in ["visit01", "visit02", "visit03", "visit04"] if v in df_all["visit"].unique()]

    # 1) histograms for size
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for visit in ordered_visits:
        dv = df_all[df_all["visit"] == visit]
        axes[0].hist(dv["area_um2"].dropna(), bins=40, alpha=0.4, label=visit)
        axes[1].hist(dv["equivalent_diameter_um"].dropna(), bins=40, alpha=0.4, label=visit)

    axes[0].set_title("Cell area distribution")
    axes[0].set_xlabel(r"Area ($\mu m^2$)")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    axes[1].set_title("Equivalent diameter distribution")
    axes[1].set_xlabel(r"Equivalent diameter ($\mu m$)")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "size_distributions_by_visit.png", dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

    # 2) boxplots
    plot_features = [
        ("area_um2", r"Area ($\mu m^2$)"),
        ("equivalent_diameter_um", r"Equivalent diameter ($\mu m$)"),
        ("circularity", "Circularity"),
        ("eccentricity", "Eccentricity"),
        ("aspect_ratio", "Aspect ratio"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for ax, (feat, ylabel) in zip(axes, plot_features):
        data = [df_all[df_all["visit"] == v][feat].dropna().values for v in ordered_visits]
        ax.boxplot(data, tick_labels=ordered_visits, showfliers=False)
        ax.set_title(feat)
        ax.set_ylabel(ylabel)

    # hide unused axis if any
    for i in range(len(plot_features), len(axes)):
        axes[i].axis("off")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "morphology_boxplots_by_visit.png", dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

    # 3) heatmap of normalized medians
    median_cols = [f"{f}_median" for f in features]
    heat = df_summary.set_index("visit")[median_cols].copy()
    heat = heat.loc[ordered_visits]

    # z-score by column
    heat_z = (heat - heat.mean(axis=0)) / heat.std(axis=0, ddof=0)
    heat_z = heat_z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(heat_z.values, aspect="auto")
    ax.set_xticks(range(len(heat_z.columns)))
    ax.set_xticklabels([c.replace("_median", "") for c in heat_z.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(heat_z.index)))
    ax.set_yticklabels(heat_z.index)
    ax.set_title("Morphology shift heatmap (z-scored visit medians)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "morphology_shift_heatmap.png", dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

    print(f"[DONE] Saved per-cell table: {PER_CELL_CSV}")
    print(f"[DONE] Saved visit summary: {PER_VISIT_SUMMARY_CSV}")
    print(f"[DONE] Saved shift table: {SHIFT_TABLE_CSV}")
    print(f"[DONE] Saved figures in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()