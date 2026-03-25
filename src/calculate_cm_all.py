#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import imageio.v3 as iio
import numpy as np
import pandas as pd
import tifffile

from skimage.measure import label, regionprops
from sklearn.mixture import GaussianMixture


# ============================================================
# CONFIG
# ============================================================

PATIENT_DIR = Path("/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449")
OUTPUT_DIR = PATIENT_DIR / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# phasor stacks: first harmonic
PHASOR_G_IDX = 1
PHASOR_S_IDX = 2

MIN_ROI_AREA = 1

FILTER_PHASOR_RANGE = False
G_MIN, G_MAX = -0.2, 1.2
S_MIN, S_MAX = -0.2, 1.2

# outputs
ROI_ALL_CSV = OUTPUT_DIR / "roi_phasor_points_all_three_types.csv"
CLUSTER_CENTERS_CSV = OUTPUT_DIR / "cluster_centers_all_three_types_by_visit.csv"
ROI_LABELED_CSV = OUTPUT_DIR / "roi_phasor_points_with_gmm_labels_all_three_types.csv"


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Case:
    patient: str
    visit: str
    mosaic_name: str
    coumarin_path: Path
    raw_path: Path
    elastin_only_path: Path
    mask_path: Path


# ============================================================
# HELPERS
# ============================================================

def infer_patient_visit_from_path(path: Path) -> Tuple[str, str]:
    path_str = str(path)

    patient_match = re.search(r"(p\d+)", path_str, re.IGNORECASE)
    visit_match = re.search(r"(visit[_-]?\d+)", path_str, re.IGNORECASE)

    patient = patient_match.group(1) if patient_match else "unknown_patient"
    visit = visit_match.group(1) if visit_match else "unknown_visit"
    return patient, visit


def n_components_for_visit(visit: str) -> int:
    return 2 if visit.lower() == "visit04" else 3


def compute_phase(g: np.ndarray, s: np.ndarray) -> np.ndarray:
    return np.arctan2(s, g)


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
    """
    For each mosaic, require:
      - _new/phasor.tif
      - phasor_uncalibrated.tif
      - phasor_uncalibrated_elastin_corr.tif
      - _new/instance_mask_filtered.*
    """
    cases: List[Case] = []

    for coumarin_path in patient_dir.rglob("phasor.tif"):
        # only use the phasor.tif inside _new
        if coumarin_path.parent.name != "_new":
            continue

        mosaic_dir = coumarin_path.parent.parent

        raw_path = mosaic_dir / "phasor_uncalibrated.tif"
        elastin_only_path = mosaic_dir / "phasor_uncalibrated_elastin_corr.tif"
        mask_path = find_mask_in_new_folder(mosaic_dir)

        if not raw_path.exists():
            print(f"[SKIP] Missing raw phasor: {raw_path}")
            continue

        if not elastin_only_path.exists():
            print(f"[SKIP] Missing elastin-corrected raw phasor: {elastin_only_path}")
            continue

        if mask_path is None:
            print(f"[SKIP] Missing instance mask in: {mosaic_dir / '_new'}")
            continue

        patient, visit = infer_patient_visit_from_path(mosaic_dir)

        cases.append(
            Case(
                patient=patient,
                visit=visit,
                mosaic_name=mosaic_dir.name,
                coumarin_path=coumarin_path,
                raw_path=raw_path,
                elastin_only_path=elastin_only_path,
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

def roi_table_from_phasor(
    phasor_path: Path,
    mask_path: Path,
    patient: str,
    visit: str,
    mosaic_name: str,
    phasor_type: str,
) -> pd.DataFrame:
    g, s = read_phasor_gs(phasor_path)
    labels = read_mask(mask_path)

    if g.shape != labels.shape or s.shape != labels.shape:
        raise ValueError(
            f"Shape mismatch: G={g.shape}, S={s.shape}, mask={labels.shape}"
        )

    rows = []
    props = regionprops(labels)

    for prop in props:
        if prop.area < MIN_ROI_AREA:
            continue

        rr = prop.coords[:, 0]
        cc = prop.coords[:, 1]

        gvals = g[rr, cc]
        svals = s[rr, cc]

        valid = np.isfinite(gvals) & np.isfinite(svals)

        if FILTER_PHASOR_RANGE:
            valid &= (
                (gvals >= G_MIN) & (gvals <= G_MAX) &
                (svals >= S_MIN) & (svals <= S_MAX)
            )

        if valid.sum() == 0:
            continue

        gvals = gvals[valid]
        svals = svals[valid]

        rows.append({
            "patient": patient,
            "visit": visit,
            "mosaic_name": mosaic_name,
            "phasor_type": phasor_type,
            "phasor_path": str(phasor_path),
            "mask_path": str(mask_path),
            "roi_label": int(prop.label),
            "area_px": int(prop.area),
            "centroid_row": float(prop.centroid[0]),
            "centroid_col": float(prop.centroid[1]),
            "g_mean": float(np.mean(gvals)),
            "s_mean": float(np.mean(svals)),
            "g_std_within_roi": float(np.std(gvals, ddof=0)),
            "s_std_within_roi": float(np.std(svals, ddof=0)),
            "n_valid_pixels": int(valid.sum()),
        })

    return pd.DataFrame(rows)


# ============================================================
# GMM + LABELS
# ============================================================

def assign_biological_labels_by_phase(
    df_roi: pd.DataFrame,
    n_components: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = df_roi[["g_mean", "s_mean"]].to_numpy()

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=0,
        n_init=10,
    )
    gmm.fit(X)

    pred = gmm.predict(X)

    df_out = df_roi.copy()
    df_out["gmm_cluster"] = pred
    df_out["phase"] = compute_phase(df_out["g_mean"].to_numpy(), df_out["s_mean"].to_numpy())

    cluster_rows = []
    for k in sorted(np.unique(pred)):
        d = df_out[df_out["gmm_cluster"] == k]
        cluster_rows.append({
            "cluster": int(k),
            "phase_mean": float(d["phase"].mean()),
            "g_mean": float(d["g_mean"].mean()),
            "s_mean": float(d["s_mean"].mean()),
            "g_std": float(d["g_mean"].std(ddof=0)),
            "s_std": float(d["s_mean"].std(ddof=0)),
            "n_rois": int(len(d)),
        })

    cluster_df = pd.DataFrame(cluster_rows).sort_values("phase_mean").reset_index(drop=True)

    if len(cluster_df) == 3:
        cluster_df.loc[0, "bio_label"] = "melanin"
        cluster_df.loc[1, "bio_label"] = "cells"
        cluster_df.loc[2, "bio_label"] = "elastin"
    elif len(cluster_df) == 2:
        cluster_df.loc[0, "bio_label"] = "cells"
        cluster_df.loc[1, "bio_label"] = "elastin"
    else:
        raise ValueError(f"Expected 2 or 3 clusters, got {len(cluster_df)}")

    cluster_to_bio = dict(zip(cluster_df["cluster"], cluster_df["bio_label"]))
    df_out["bio_label"] = df_out["gmm_cluster"].map(cluster_to_bio)

    return df_out, cluster_df


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    cases = collect_cases(PATIENT_DIR)

    if not cases:
        raise RuntimeError("No valid mosaics found with the three phasor types and instance mask.")

    print(f"[INFO] Found {len(cases)} mosaics")
    for c in cases:
        print(f"  - {c.visit} | {c.mosaic_name}")

    all_roi_tables = []

    for case in cases:
        try:
            dfs = [
                roi_table_from_phasor(
                    case.coumarin_path,
                    case.mask_path,
                    patient=case.patient,
                    visit=case.visit,
                    mosaic_name=case.mosaic_name,
                    phasor_type="coumarin_calibrated",
                ),
                roi_table_from_phasor(
                    case.raw_path,
                    case.mask_path,
                    patient=case.patient,
                    visit=case.visit,
                    mosaic_name=case.mosaic_name,
                    phasor_type="uncalibrated",
                ),
                roi_table_from_phasor(
                    case.elastin_only_path,
                    case.mask_path,
                    patient=case.patient,
                    visit=case.visit,
                    mosaic_name=case.mosaic_name,
                    phasor_type="uncalibrated_elastin_corr",
                ),
            ]

            for df in dfs:
                if len(df) > 0:
                    all_roi_tables.append(df)

            print(f"[OK] {case.visit} | {case.mosaic_name}")

        except Exception as e:
            print(f"[ERROR] {case.visit} | {case.mosaic_name}: {e}")

    if not all_roi_tables:
        raise RuntimeError("No ROI tables could be generated.")

    df_all = pd.concat(all_roi_tables, ignore_index=True)
    df_all.to_csv(ROI_ALL_CSV, index=False)
    print(f"[INFO] Saved ROI table: {ROI_ALL_CSV}")

    all_cluster_centers = []
    all_roi_labeled = []

    for visit in sorted(df_all["visit"].unique()):
        n_components = n_components_for_visit(visit)

        for phasor_type in [
            "uncalibrated",
            "uncalibrated_elastin_corr",
            "coumarin_calibrated",
        ]:
            df_subset = df_all[
                (df_all["visit"] == visit) &
                (df_all["phasor_type"] == phasor_type)
            ].copy()

            if len(df_subset) < 2:
                print(f"[WARN] Too few ROIs for {visit} | {phasor_type}, skipping GMM")
                continue

            try:
                df_labeled, cluster_df = assign_biological_labels_by_phase(
                    df_subset,
                    n_components=n_components,
                )

                df_labeled["visit"] = visit
                df_labeled["phasor_type"] = phasor_type

                cluster_df["visit"] = visit
                cluster_df["phasor_type"] = phasor_type

                all_roi_labeled.append(df_labeled)
                all_cluster_centers.append(cluster_df)

                print(f"[OK] GMM: {visit} | {phasor_type} | n_rois={len(df_subset)}")

            except Exception as e:
                print(f"[ERROR] GMM failed for {visit} | {phasor_type}: {e}")

    if all_roi_labeled:
        df_roi_labeled = pd.concat(all_roi_labeled, ignore_index=True)
        df_roi_labeled.to_csv(ROI_LABELED_CSV, index=False)
        print(f"[INFO] Saved labeled ROI table: {ROI_LABELED_CSV}")

    if all_cluster_centers:
        df_cluster_centers = pd.concat(all_cluster_centers, ignore_index=True)
        df_cluster_centers = df_cluster_centers[
            ["visit", "phasor_type", "cluster", "bio_label", "phase_mean", "g_mean", "s_mean", "g_std", "s_std", "n_rois"]
        ]
        df_cluster_centers.to_csv(CLUSTER_CENTERS_CSV, index=False)
        print(f"[INFO] Saved cluster centers: {CLUSTER_CENTERS_CSV}")

    print("[DONE]")


if __name__ == "__main__":
    main()