#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot tissue phasor centroids by visit for raw, calibrated, and corrected phasors.

This QC script uses tissue masks generated in segmentation_area_phasor/ to compute
the center of mass of:

    1. elastin
    2. cells
    3. melanin

for each visit and phasor state:

    1. raw
    2. coumarin-calibrated
    3. elastin-corrected

It plots one separate figure per tissue and detector channel.

Markers:
    raw        = x
    calibrated = +
    corrected  = o

Outputs:
    analysis/elastin_correction_qc/
        tissue_centroids_raw_calibrated_corrected.csv
        elastin_centroids_green_raw_calibrated_corrected.png
        elastin_centroids_blue_raw_calibrated_corrected.png
        cells_centroids_green_raw_calibrated_corrected.png
        cells_centroids_blue_raw_calibrated_corrected.png
        melanin_centroids_green_raw_calibrated_corrected.png
        melanin_centroids_blue_raw_calibrated_corrected.png
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from phasorpy.plot import PhasorPlot
from skimage.measure import label, regionprops

# ============================================================
# CONFIG
# ============================================================

PATIENT_DIR = Path(
    "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449"
).expanduser()

PHASOR_SUBDIR = "phasor"
SEGMENTATION_SUBDIR = "segmentation_area_phasor"

RAW_NAME = "phasor_raw_green_blue_mosaic.tif"
CAL_NAME = "phasor_calibrated_green_blue_mosaic.tif"
CORR_NAME = "phasor_raw_green_blue_mosaic_elastin_corrected.tif"

TISSUE_MASK_SUFFIXES = {
    "elastin": "_elastin_mask.tif",
    "cells": "_cell_mask_final.tif",
    "melanin": "_melanin_mask.tif",
}

OUTPUT_DIR = PATIENT_DIR / "analysis" / "elastin_correction_qc"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUTPUT_DIR / "tissue_centroids_raw_calibrated_corrected.csv"

FIG_DPI = 600
FREQUENCY_MHZ = 80.0

G_GREEN_IDX = 1
S_GREEN_IDX = 2
G_BLUE_IDX = 3
S_BLUE_IDX = 4

MIN_ROI_AREA = 1
MIN_VALID_PIXELS = 1

FILTER_PHASOR_RANGE = False
G_MIN, G_MAX = -1.5, 1.5
S_MIN, S_MAX = -1.5, 1.5

X_LIMITS = None
Y_LIMITS = None


# ============================================================
# DATA STRUCTURES
# ============================================================


@dataclass
class Case:
    visit: str
    mosaic_name: str
    mosaic_dir: Path
    tissue: str
    mask_path: Path
    raw_path: Path
    calibrated_path: Path
    corrected_path: Path


# ============================================================
# HELPERS
# ============================================================


def natural_key(path: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def infer_visit_from_path(path: Path) -> str:
    match = re.search(r"(visit[_-]?\d+)", str(path), re.IGNORECASE)
    return match.group(1) if match else "unknown_visit"


def find_tissue_mask(mosaic_dir: Path, tissue: str) -> Optional[Path]:
    seg_dir = mosaic_dir / SEGMENTATION_SUBDIR

    if not seg_dir.exists():
        return None

    suffix = TISSUE_MASK_SUFFIXES[tissue]

    candidates = sorted(
        list(seg_dir.glob(f"*{suffix}"))
        + list(seg_dir.glob(f"*{suffix.replace('.tif', '.tiff')}")),
        key=natural_key,
    )

    if len(candidates) == 0:
        return None

    return candidates[0]


def collect_cases(patient_dir: Path) -> list[Case]:
    cases = []

    mosaic_dirs = sorted(
        [p for p in patient_dir.glob("visit*/Mosaic*") if p.is_dir()],
        key=natural_key,
    )

    for mosaic_dir in mosaic_dirs:
        phasor_dir = mosaic_dir / PHASOR_SUBDIR

        raw_path = phasor_dir / RAW_NAME
        calibrated_path = phasor_dir / CAL_NAME
        corrected_path = phasor_dir / CORR_NAME

        if not raw_path.exists():
            print(f"[SKIP] Missing raw phasor: {raw_path}")
            continue

        if not calibrated_path.exists():
            print(f"[SKIP] Missing calibrated phasor: {calibrated_path}")
            continue

        if not corrected_path.exists():
            print(f"[SKIP] Missing corrected phasor: {corrected_path}")
            continue

        for tissue in TISSUE_MASK_SUFFIXES:
            mask_path = find_tissue_mask(mosaic_dir, tissue)

            if mask_path is None:
                print(f"[SKIP] Missing {tissue} mask: {mosaic_dir}")
                continue

            cases.append(
                Case(
                    visit=infer_visit_from_path(mosaic_dir),
                    mosaic_name=mosaic_dir.name,
                    mosaic_dir=mosaic_dir,
                    tissue=tissue,
                    mask_path=mask_path,
                    raw_path=raw_path,
                    calibrated_path=calibrated_path,
                    corrected_path=corrected_path,
                )
            )

    return cases


def read_mask(mask_path: Path) -> np.ndarray:
    if mask_path.suffix.lower() in {".tif", ".tiff"}:
        mask = tifffile.imread(mask_path)
    else:
        mask = iio.imread(mask_path)

    mask = np.asarray(mask).squeeze()

    if mask.ndim == 3:
        if mask.shape[-1] in (3, 4):
            mask = mask[..., :3].max(axis=-1)
        else:
            raise ValueError(f"Unsupported mask shape {mask.shape} in {mask_path}")

    if mask.dtype == bool:
        return label(mask).astype(np.int32)

    unique_vals = np.unique(mask)

    if set(unique_vals.tolist()).issubset({0, 1, 255}):
        return label(mask > 0).astype(np.int32)

    return mask.astype(np.int32)


def read_phasor(path: Path) -> np.ndarray:
    stack = tifffile.imread(path)
    stack = np.asarray(stack).squeeze()

    if stack.ndim != 3:
        raise ValueError(f"Expected CYX phasor stack, got {stack.shape} in {path}")

    if stack.shape[0] < 5:
        raise ValueError(f"Expected at least 5 planes, got {stack.shape[0]} in {path}")

    return stack.astype(np.float32, copy=False)


def compute_roi_mean_points(
    stack: np.ndarray,
    labels: np.ndarray,
    *,
    g_idx: int,
    s_idx: int,
) -> list[dict]:
    g = stack[g_idx].astype(np.float64)
    s = stack[s_idx].astype(np.float64)

    if g.shape != labels.shape:
        raise ValueError(f"Shape mismatch: phasor={g.shape}, mask={labels.shape}")

    rows = []

    for prop in regionprops(labels):
        if prop.area < MIN_ROI_AREA:
            continue

        rr = prop.coords[:, 0]
        cc = prop.coords[:, 1]

        gvals = g[rr, cc]
        svals = s[rr, cc]

        valid = np.isfinite(gvals) & np.isfinite(svals)

        if FILTER_PHASOR_RANGE:
            valid &= (
                (gvals >= G_MIN)
                & (gvals <= G_MAX)
                & (svals >= S_MIN)
                & (svals <= S_MAX)
            )

        if valid.sum() < MIN_VALID_PIXELS:
            continue

        rows.append(
            {
                "g_roi_mean": float(np.mean(gvals[valid])),
                "s_roi_mean": float(np.mean(svals[valid])),
                "area_px": int(prop.area),
                "n_valid_pixels": int(valid.sum()),
            }
        )

    return rows


def extract_case_rows(case: Case) -> list[dict]:
    labels = read_mask(case.mask_path)

    phasor_paths = {
        "raw": case.raw_path,
        "calibrated": case.calibrated_path,
        "corrected": case.corrected_path,
    }

    channel_info = {
        "green": (G_GREEN_IDX, S_GREEN_IDX),
        "blue": (G_BLUE_IDX, S_BLUE_IDX),
    }

    rows = []

    for phasor_state, path in phasor_paths.items():
        stack = read_phasor(path)

        for channel, (g_idx, s_idx) in channel_info.items():
            roi_rows = compute_roi_mean_points(
                stack,
                labels,
                g_idx=g_idx,
                s_idx=s_idx,
            )

            for r in roi_rows:
                r.update(
                    {
                        "visit": case.visit,
                        "mosaic_name": case.mosaic_name,
                        "tissue": case.tissue,
                        "phasor_state": phasor_state,
                        "channel": channel,
                        "phasor_path": str(path),
                        "mask_path": str(case.mask_path),
                    }
                )
                rows.append(r)

    return rows


def summarize_centroids(df_roi: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df_roi.groupby(["tissue", "visit", "phasor_state", "channel"])
        .agg(
            g_centroid=("g_roi_mean", "mean"),
            s_centroid=("s_roi_mean", "mean"),
            g_std=("g_roi_mean", "std"),
            s_std=("s_roi_mean", "std"),
            n_rois=("g_roi_mean", "size"),
            total_area_px=("area_px", "sum"),
        )
        .reset_index()
    )

    summary["g_std"] = summary["g_std"].fillna(0.0)
    summary["s_std"] = summary["s_std"].fillna(0.0)

    global_rows = []

    for (tissue, phasor_state, channel), d in summary.groupby(
        ["tissue", "phasor_state", "channel"]
    ):
        global_rows.append(
            {
                "tissue": tissue,
                "visit": "global_mean",
                "phasor_state": phasor_state,
                "channel": channel,
                "g_centroid": float(d["g_centroid"].mean()),
                "s_centroid": float(d["s_centroid"].mean()),
                "g_std": float(d["g_centroid"].std(ddof=0)),
                "s_std": float(d["s_centroid"].std(ddof=0)),
                "n_rois": int(d["n_rois"].sum()),
                "total_area_px": int(d["total_area_px"].sum()),
            }
        )

    return pd.concat([summary, pd.DataFrame(global_rows)], ignore_index=True)


def plot_channel_centroids(
    summary: pd.DataFrame,
    *,
    tissue: str,
    channel: str,
    out_path: Path,
) -> None:
    df = summary[(summary["tissue"] == tissue) & (summary["channel"] == channel)].copy()

    if df.empty:
        print(f"[SKIP] No summary rows for {tissue} | {channel}")
        return

    visits = sorted(
        [v for v in df["visit"].unique() if v != "global_mean"],
        key=lambda x: (
            int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else 999
        ),
    )

    visit_colors = {
        "visit01": "tab:blue",
        "visit02": "tab:orange",
        "visit03": "tab:green",
        "visit04": "tab:red",
    }

    state_styles = {
        "raw": {
            "marker": "x",
            "alpha": 0.65,
            "size": 120,
            "linewidth": 2.0,
            "label_suffix": "raw",
        },
        "calibrated": {
            "marker": "+",
            "alpha": 0.85,
            "size": 140,
            "linewidth": 2.2,
            "label_suffix": "calibrated",
        },
        "corrected": {
            "marker": "o",
            "alpha": 1.00,
            "size": 95,
            "linewidth": 0.8,
            "label_suffix": "corrected",
        },
    }

    fig, ax = plt.subplots(figsize=(8, 7))

    plot = PhasorPlot(
        ax=ax,
        frequency=FREQUENCY_MHZ,
        title=f"{tissue.capitalize()} centroids - {channel} detector",
    )
    plot.semicircle()

    for visit in visits:
        color = visit_colors.get(visit, "gray")

        for state, style in state_styles.items():
            row = df[(df["visit"] == visit) & (df["phasor_state"] == state)]

            if row.empty:
                continue

            row = row.iloc[0]

            if state == "corrected":
                ax.scatter(
                    row["g_centroid"],
                    row["s_centroid"],
                    s=style["size"],
                    marker=style["marker"],
                    color=color,
                    alpha=style["alpha"],
                    edgecolor="black",
                    linewidth=style["linewidth"],
                    label=f"{visit} | {style['label_suffix']}",
                )
            else:
                ax.scatter(
                    row["g_centroid"],
                    row["s_centroid"],
                    s=style["size"],
                    marker=style["marker"],
                    color=color,
                    alpha=style["alpha"],
                    linewidth=style["linewidth"],
                    label=f"{visit} | {style['label_suffix']}",
                )

    for state, style in state_styles.items():
        row = df[(df["visit"] == "global_mean") & (df["phasor_state"] == state)]

        if row.empty:
            continue

        row = row.iloc[0]

        if state == "corrected":
            ax.scatter(
                row["g_centroid"],
                row["s_centroid"],
                s=260,
                marker="o",
                color="black",
                alpha=1.0,
                edgecolor="white",
                linewidth=1.2,
                label=f"global mean | {state}",
            )
        else:
            ax.scatter(
                row["g_centroid"],
                row["s_centroid"],
                s=240,
                marker=style["marker"],
                color="black",
                alpha=style["alpha"],
                linewidth=style["linewidth"],
                label=f"global mean | {state}",
            )

    if X_LIMITS is not None:
        ax.set_xlim(*X_LIMITS)

    if Y_LIMITS is not None:
        ax.set_ylim(*Y_LIMITS)

    ax.legend(
        fontsize=7,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
    )

    fig.tight_layout()
    fig.savefig(
        out_path,
        dpi=FIG_DPI,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    cases = collect_cases(PATIENT_DIR)

    if not cases:
        raise RuntimeError("No valid tissue-mask cases found.")

    print(f"[INFO] Found {len(cases)} valid tissue-mask cases")

    all_rows = []

    for case in cases:
        try:
            rows = extract_case_rows(case)
            all_rows.extend(rows)
            print(
                f"[OK] {case.visit} | {case.mosaic_name} | "
                f"{case.tissue} | rows = {len(rows)}"
            )

        except Exception as e:
            print(f"[ERROR] {case.visit} | {case.mosaic_name} | {case.tissue}: {e}")

    if not all_rows:
        raise RuntimeError("No tissue ROI rows extracted.")

    df_roi = pd.DataFrame(all_rows)

    summary = summarize_centroids(df_roi)
    summary.to_csv(OUT_CSV, index=False)

    print("\n[INFO] Saved centroid table:")
    print(f"       {OUT_CSV}")

    for tissue in TISSUE_MASK_SUFFIXES:
        for channel in ["green", "blue"]:
            out_path = (
                OUTPUT_DIR
                / f"{tissue}_centroids_{channel}_raw_calibrated_corrected.png"
            )

            plot_channel_centroids(
                summary,
                tissue=tissue,
                channel=channel,
                out_path=out_path,
            )

            print(f"[INFO] Saved QC plot: {out_path}")

    print("\nCentroid summary:")
    print(
        summary[
            [
                "tissue",
                "visit",
                "phasor_state",
                "channel",
                "g_centroid",
                "s_centroid",
                "n_rois",
            ]
        ].to_string(index=False)
    )

    print("\n[DONE]")


if __name__ == "__main__":
    main()
