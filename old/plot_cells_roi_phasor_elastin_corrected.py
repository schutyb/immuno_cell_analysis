#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

from skimage.measure import label, regionprops

from phasorpy.plot import plot_phasor


# ============================================================
# CONFIG
# ============================================================

PATIENT_DIR = Path("/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449")
ANALYSIS_DIR = PATIENT_DIR / "analysis"

# use the ROI labels table already generated from the 3-case comparison
ROI_LABELS_CSV = ANALYSIS_DIR / "roi_phasor_points_with_gmm_labels_all_three_types.csv"

OUTPUT_DIR = ANALYSIS_DIR / "cells_roi_phasor_elastin_corrected"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PHASOR_TYPE_TO_USE = "uncalibrated_elastin_corr"
BIO_LABEL_TO_KEEP = "cells"

PHASOR_G_IDX = 1
PHASOR_S_IDX = 2
MIN_ROI_AREA = 1
PHASOR_FREQUENCY = 80.0
SHOW_PLOTS = True

CELL_COLOR = "red"

VISIT_ORDER = ["visit01", "visit02", "visit03", "visit04"]


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Case:
    patient: str
    visit: str
    mosaic_name: str
    phasor_path: Path
    mask_path: Path


# ============================================================
# HELPERS
# ============================================================

def infer_patient_visit_from_path(path: Path) -> Tuple[str, str]:
    path_str = str(path)

    patient_match = re.search(r"(p\d+)", path_str, re.IGNORECASE)
    visit_match = re.search(r"(visit[_-]?\d+)", path_str, re.IGNORECASE)

    patient = patient_match.group(1) if patient_match else "unknown_patient"
    visit = visit_match.group(1).lower() if visit_match else "unknown_visit"
    return patient, visit


def short_visit_label(visit: str) -> str:
    visit = str(visit).lower()
    if visit.startswith("visit"):
        try:
            return f"Visit {int(visit.replace('visit', '')):02d}"
        except ValueError:
            return visit
    return visit


# ============================================================
# SEARCH
# ============================================================

def find_mask_in_new_folder(mosaic_dir: Path) -> Optional[Path]:
    candidates = [
        mosaic_dir / "_new" / "instance_mask_filtered.tif",
        mosaic_dir / "_new" / "instance_mask_filtered.tiff",
        mosaic_dir / "_new" / "instance_mask_filtered.png",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def collect_cases(patient_dir: Path) -> List[Case]:
    cases: List[Case] = []

    for phasor_path in patient_dir.rglob("phasor_uncalibrated_elastin_corr.tif"):
        mosaic_dir = phasor_path.parent
        mask_path = find_mask_in_new_folder(mosaic_dir)

        if mask_path is None:
            print(f"[SKIP] Missing instance mask in: {mosaic_dir / '_new'}")
            continue

        patient, visit = infer_patient_visit_from_path(mosaic_dir)

        cases.append(
            Case(
                patient=patient,
                visit=visit,
                mosaic_name=mosaic_dir.name,
                phasor_path=phasor_path,
                mask_path=mask_path,
            )
        )

    return sorted(cases, key=lambda c: (c.patient, c.visit, c.mosaic_name))


# ============================================================
# READERS
# ============================================================

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


def read_phasor_gs(phasor_path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = tifffile.imread(phasor_path)
    arr = np.asarray(arr).squeeze()

    if arr.ndim != 3:
        raise ValueError(f"Unexpected phasor shape: {arr.shape} in {phasor_path}")

    g = arr[PHASOR_G_IDX].astype(np.float64)
    s = arr[PHASOR_S_IDX].astype(np.float64)
    return g, s


# ============================================================
# ROI TABLE
# ============================================================

def roi_table_from_case(
    case: Case,
    df_labels_all: pd.DataFrame,
) -> pd.DataFrame:
    g, s = read_phasor_gs(case.phasor_path)
    labels = read_mask(case.mask_path)

    if g.shape != labels.shape or s.shape != labels.shape:
        raise ValueError(
            f"Shape mismatch: G={g.shape}, S={s.shape}, mask={labels.shape}"
        )

    df_keep = df_labels_all[
        (df_labels_all["visit"] == case.visit) &
        (df_labels_all["mosaic_name"] == case.mosaic_name) &
        (df_labels_all["phasor_type"] == PHASOR_TYPE_TO_USE) &
        (df_labels_all["bio_label"].str.lower() == BIO_LABEL_TO_KEEP)
    ].copy()

    keep_labels = set(df_keep["roi_label"].astype(int).tolist())
    if not keep_labels:
        return pd.DataFrame()

    rows = []
    props = regionprops(labels)

    for prop in props:
        roi_label = int(prop.label)
        if roi_label not in keep_labels:
            continue

        if prop.area < MIN_ROI_AREA:
            continue

        rr = prop.coords[:, 0]
        cc = prop.coords[:, 1]

        gvals = g[rr, cc]
        svals = s[rr, cc]

        valid = np.isfinite(gvals) & np.isfinite(svals)
        if valid.sum() == 0:
            continue

        gvals = gvals[valid]
        svals = svals[valid]

        rows.append({
            "patient": case.patient,
            "visit": case.visit,
            "mosaic_name": case.mosaic_name,
            "roi_label": roi_label,
            "g_mean": float(np.mean(gvals)),
            "s_mean": float(np.mean(svals)),
            "n_valid_pixels": int(valid.sum()),
        })

    return pd.DataFrame(rows)


# ============================================================
# PLOTTING
# ============================================================

def plot_visit_panel(ax, df_visit: pd.DataFrame, visit: str) -> None:
    title = short_visit_label(visit)

    if df_visit.empty:
        plot_phasor(
            np.array([0.5]),
            np.array([0.0]),
            style="plot",
            marker="",
            linestyle="",
            frequency=PHASOR_FREQUENCY,
            ax=ax,
            title=title,
            show=False,
        )
        ax.text(0.5, 0.35, "No cell ROIs", ha="center", va="center", fontsize=12)
        return

    plot_phasor(
        df_visit["g_mean"].to_numpy(),
        df_visit["s_mean"].to_numpy(),
        style="plot",
        marker=".",
        linestyle="",
        color=CELL_COLOR,
        label=f"Cells (n={len(df_visit)})",
        frequency=PHASOR_FREQUENCY,
        ax=ax,
        title=title,
        show=False,
    )

    # centroid of all cell ROIs in the visit
    ax.scatter(
        df_visit["g_mean"].mean(),
        df_visit["s_mean"].mean(),
        s=160,
        c=CELL_COLOR,
        edgecolors="black",
        linewidths=1.0,
        marker="X",
        zorder=10,
        label="Cell centroid",
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 0.7)
    ax.legend(fontsize=8)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not ROI_LABELS_CSV.exists():
        raise FileNotFoundError(f"Missing ROI labels CSV: {ROI_LABELS_CSV}")

    df_labels = pd.read_csv(ROI_LABELS_CSV)
    required = {"visit", "mosaic_name", "roi_label", "bio_label", "phasor_type"}
    missing = required - set(df_labels.columns)
    if missing:
        raise ValueError(f"Missing columns in ROI labels CSV: {sorted(missing)}")

    df_labels["visit"] = df_labels["visit"].astype(str).str.lower()
    df_labels["bio_label"] = df_labels["bio_label"].astype(str).str.lower()
    df_labels["mosaic_name"] = df_labels["mosaic_name"].astype(str)

    cases = collect_cases(PATIENT_DIR)

    if not cases:
        raise RuntimeError("No valid mosaics found with elastin-corrected phasor + instance mask.")

    print(f"[INFO] Found {len(cases)} mosaics")

    all_roi_tables = []
    for case in cases:
        try:
            df_case = roi_table_from_case(case, df_labels)
            if len(df_case) == 0:
                print(f"[WARN] No cell ROIs for {case.visit} | {case.mosaic_name}")
                continue

            all_roi_tables.append(df_case)
            print(f"[OK] {case.visit} | {case.mosaic_name} | cell ROIs={len(df_case)}")

        except Exception as e:
            print(f"[ERROR] {case.visit} | {case.mosaic_name}: {e}")

    if not all_roi_tables:
        raise RuntimeError("No cell ROI tables could be generated.")

    df_all = pd.concat(all_roi_tables, ignore_index=True)
    out_csv = OUTPUT_DIR / "cells_roi_phasor_elastin_corrected.csv"
    df_all.to_csv(out_csv, index=False)

    # figure with 4 panels (one per visit)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, visit in zip(axes, VISIT_ORDER):
        df_visit = df_all[df_all["visit"] == visit].copy()
        plot_visit_panel(ax, df_visit, visit)

    fig.suptitle("Cell ROI phasor plots - elastin-corrected raw phasor", fontsize=14)
    fig.tight_layout()

    out_png = OUTPUT_DIR / "cells_roi_phasor_elastin_corrected_four_visits.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

    print(f"[DONE] Saved ROI table: {out_csv}")
    print(f"[DONE] Saved figure: {out_png}")


if __name__ == "__main__":
    main()