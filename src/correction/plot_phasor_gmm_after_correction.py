#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot ROI-level phasor distributions after elastin correction.

This script:
    1. Reads the elastin-corrected phasor mosaics.
    2. Reads ROI masks:
        - elastin
        - cells
        - cell_low_confidence
        - melanin (if present)
    3. Computes ROI-level mean phasor coordinates.
    4. Generates visit-level phasor QC plots.

The plots allow visual inspection of whether the final corrected phasor
representation produces biologically consistent ROI clusters across visits.

Input phasor:
    phasor/phasor_raw_green_blue_mosaic_elastin_corrected.tif

Input masks:
    segmentation_area_phasor/*_elastin_mask.tif
    segmentation_area_phasor/*_cell_mask_final.tif
    segmentation_area_phasor/*_cell_low_confidence_mask.tif
    segmentation_area_phasor/*_melanin_mask.tif

Output:
    analysis/final_roi_phasor_qc/
        roi_phasor_corrected_all_visits.csv
        visitXX_green_roi_phasor_corrected.png
        visitXX_blue_roi_phasor_corrected.png
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

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

CORRECTED_PHASOR_NAME = "phasor_raw_green_blue_mosaic_elastin_corrected.tif"

OUTPUT_DIR = PATIENT_DIR / "analysis" / "final_roi_phasor_qc"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUTPUT_DIR / "roi_phasor_corrected_all_visits.csv"

# Masks
MASK_SUFFIXES = {
    "elastin": "_elastin_mask.tif",
    "cells": "_cell_mask_final.tif",
    "cell_low_confidence": "_cell_low_confidence_mask.tif",
    "melanin": "_melanin_mask.tif",
}

# Phasor planes
G_GREEN_IDX = 1
S_GREEN_IDX = 2
G_BLUE_IDX = 3
S_BLUE_IDX = 4

FIG_DPI = 600
FREQUENCY_MHZ = 80.0

MIN_ROI_AREA = 1
MIN_VALID_PIXELS = 1

BIN_SIZE = 1

# ============================================================
# HELPERS
# ============================================================


def natural_key(path: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def natural_sort_key_text(text):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(text))]


def infer_visit(path: Path) -> str:
    match = re.search(r"(visit[_-]?\d+)", str(path), re.IGNORECASE)

    if match:
        return match.group(1)

    return "unknown_visit"


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

    return stack.astype(np.float32, copy=False)


def get_valid_points(g, s):
    valid = np.isfinite(g) & np.isfinite(s)

    return g[valid].ravel(), s[valid].ravel()


# ============================================================
# DATA STRUCTURE
# ============================================================


@dataclass
class Case:
    visit: str
    mosaic_name: str
    corrected_phasor_path: Path
    mask_paths: dict[str, Path]


# ============================================================
# SEARCH
# ============================================================


def collect_cases(patient_dir: Path) -> list[Case]:
    cases = []

    mosaic_dirs = sorted(
        [p for p in patient_dir.glob("visit*/Mosaic*") if p.is_dir()],
        key=natural_key,
    )

    for mosaic_dir in mosaic_dirs:
        phasor_path = mosaic_dir / PHASOR_SUBDIR / CORRECTED_PHASOR_NAME

        if not phasor_path.exists():
            print(f"[SKIP] Missing corrected phasor: {phasor_path}")
            continue

        seg_dir = mosaic_dir / SEGMENTATION_SUBDIR

        if not seg_dir.exists():
            print(f"[SKIP] Missing segmentation folder: {seg_dir}")
            continue

        mask_paths = {}

        for label_name, suffix in MASK_SUFFIXES.items():
            candidates = sorted(
                seg_dir.glob(f"*{suffix}"),
                key=natural_key,
            )

            if len(candidates) > 0:
                mask_paths[label_name] = candidates[0]

        if len(mask_paths) == 0:
            print(f"[SKIP] No masks found: {seg_dir}")
            continue

        cases.append(
            Case(
                visit=infer_visit(mosaic_dir),
                mosaic_name=mosaic_dir.name,
                corrected_phasor_path=phasor_path,
                mask_paths=mask_paths,
            )
        )

    return cases


# ============================================================
# ROI EXTRACTION
# ============================================================


def extract_roi_rows(
    case: Case,
    mask_label: str,
    mask_path: Path,
) -> list[dict]:
    stack = read_phasor(case.corrected_phasor_path)
    labels = read_mask(mask_path)

    rows = []

    channel_info = {
        "green": (G_GREEN_IDX, S_GREEN_IDX),
        "blue": (G_BLUE_IDX, S_BLUE_IDX),
    }

    for channel, (g_idx, s_idx) in channel_info.items():
        g = stack[g_idx].astype(np.float64)
        s = stack[s_idx].astype(np.float64)

        for prop in regionprops(labels):
            if prop.area < MIN_ROI_AREA:
                continue

            rr = prop.coords[:, 0]
            cc = prop.coords[:, 1]

            gvals = g[rr, cc]
            svals = s[rr, cc]

            valid = np.isfinite(gvals) & np.isfinite(svals)

            if valid.sum() < MIN_VALID_PIXELS:
                continue

            rows.append(
                {
                    "visit": case.visit,
                    "mosaic_name": case.mosaic_name,
                    "mask_type": mask_label,
                    "channel": channel,
                    "roi_label": int(prop.label),
                    "area_px": int(prop.area),
                    "g_mean": float(np.mean(gvals[valid])),
                    "s_mean": float(np.mean(svals[valid])),
                    "n_valid_pixels": int(valid.sum()),
                }
            )

    return rows


# ============================================================
# PLOTTING
# ============================================================


def plot_visit_channel(
    df: pd.DataFrame,
    *,
    visit: str,
    channel: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))

    plot = PhasorPlot(
        ax=ax,
        frequency=FREQUENCY_MHZ,
        title=f"{visit} | {channel} detector",
    )

    plot.semicircle()

    style_map = {
        "elastin": {
            "color": "tab:green",
            "label": "Elastin",
            "alpha": 0.8,
        },
        "cells": {
            "color": "tab:red",
            "label": "Cells",
            "alpha": 0.8,
        },
        "cell_low_confidence": {
            "color": "gold",
            "label": "Cell low confidence",
            "alpha": 0.6,
        },
        "melanin": {
            "color": "black",
            "label": "Melanin",
            "alpha": 0.8,
        },
    }

    for mask_type, style in style_map.items():
        d = df[df["mask_type"] == mask_type]

        if len(d) == 0:
            continue

        gvals = d["g_mean"].to_numpy()
        svals = d["s_mean"].to_numpy()

        ax.scatter(
            gvals,
            svals,
            s=10,
            color=style["color"],
            alpha=style["alpha"],
            linewidth=0,
            label=style["label"],
        )

    ax.legend(fontsize=8)

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
        raise RuntimeError("No valid mosaics found.")

    print(f"[INFO] Found {len(cases)} valid mosaics")

    all_rows = []

    for case in cases:
        for mask_label, mask_path in case.mask_paths.items():
            try:
                rows = extract_roi_rows(
                    case,
                    mask_label,
                    mask_path,
                )

                all_rows.extend(rows)

                print(f"[OK] {case.visit} | " f"{case.mosaic_name} | " f"{mask_label}")

            except Exception as e:
                print(
                    f"[ERROR] {case.visit} | "
                    f"{case.mosaic_name} | "
                    f"{mask_label}: {e}"
                )

    if len(all_rows) == 0:
        raise RuntimeError("No ROI phasor rows extracted.")

    df = pd.DataFrame(all_rows)

    df.to_csv(OUT_CSV, index=False)

    print("\n[INFO] Saved CSV:")
    print(f"       {OUT_CSV}")

    for visit in sorted(
        df["visit"].unique(),
        key=natural_sort_key_text,
    ):
        for channel in ["green", "blue"]:
            d = df[(df["visit"] == visit) & (df["channel"] == channel)]

            if len(d) == 0:
                continue

            out_path = OUTPUT_DIR / f"{visit}_{channel}_roi_phasor_corrected.png"

            plot_visit_channel(
                d,
                visit=visit,
                channel=channel,
                out_path=out_path,
            )

            print(f"[OK] Saved: {out_path}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
