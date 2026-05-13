#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute elastin-based correction parameters from raw phasor mosaics.

This script computes correction parameters to transform raw phasor coordinates
into the global calibrated elastin reference frame.

For each visit and detector channel:

    source = visit-level elastin centroid from raw phasor
    target = global elastin centroid from calibrated phasor

Then:

    dphi = phi_target_global_calibrated - phi_source_raw_visit
    mod_scale = mod_target_global_calibrated / mod_source_raw_visit

The correction is computed independently for:
    - green detector phasor: planes 1, 2
    - blue detector phasor: planes 3, 4

Input phasor planes:
    0 = DC
    1 = G green
    2 = S green
    3 = G blue
    4 = S blue

Outputs:
    analysis/elastin_correction/elastin_roi_raw_and_calibrated_phasor_all_visits.csv
    analysis/elastin_correction/elastin_correction_parameters_to_calibrated_global.csv
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import imageio.v3 as iio
import numpy as np
import pandas as pd
import tifffile
from skimage.measure import label, regionprops

# ============================================================
# CONFIG
# ============================================================

PATIENT_DIR = Path(
    "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449"
).expanduser()

PHASOR_SUBDIR = "phasor"
RAW_PHASOR_NAME = "phasor_raw_green_blue_mosaic.tif"
CALIBRATED_PHASOR_NAME = "phasor_calibrated_green_blue_mosaic.tif"

SEGMENTATION_SUBDIR = "segmentation_area_phasor"
ELASTIN_MASK_SUFFIX = "_elastin_mask.tif"

OUTPUT_DIR = PATIENT_DIR / "analysis" / "elastin_correction"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ROI_OUTPUT_CSV = OUTPUT_DIR / "elastin_roi_raw_and_calibrated_phasor_all_visits.csv"
PARAMS_OUTPUT_CSV = (
    OUTPUT_DIR / "elastin_correction_parameters_to_calibrated_global.csv"
)

# Phasor planes
DC_IDX = 0
G_GREEN_IDX = 1
S_GREEN_IDX = 2
G_BLUE_IDX = 3
S_BLUE_IDX = 4

# ROI filters
MIN_ROI_AREA = 1
MIN_VALID_PIXELS = 1

# Optional phasor range filtering
FILTER_PHASOR_RANGE = False
G_MIN, G_MAX = -1.5, 1.5
S_MIN, S_MAX = -1.5, 1.5


# ============================================================
# DATA STRUCTURES
# ============================================================


@dataclass
class Case:
    patient: str
    visit: str
    mosaic_name: str
    mosaic_dir: Path
    raw_phasor_path: Path
    calibrated_phasor_path: Path
    elastin_mask_path: Path


# ============================================================
# BASIC HELPERS
# ============================================================


def natural_key(path: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def infer_patient_visit_from_path(path: Path) -> tuple[str, str]:
    path_str = str(path)

    patient_match = re.search(r"(p\d+)", path_str, re.IGNORECASE)
    visit_match = re.search(r"(visit[_-]?\d+)", path_str, re.IGNORECASE)

    patient = patient_match.group(1) if patient_match else "unknown_patient"
    visit = visit_match.group(1) if visit_match else "unknown_visit"

    return patient, visit


def phasor_to_polar(g: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(g, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)

    mod = np.sqrt(g**2 + s**2)
    phi = np.arctan2(s, g)

    return mod, phi


def circular_mean(angles_rad: np.ndarray) -> float:
    angles_rad = np.asarray(angles_rad, dtype=np.float64)

    if angles_rad.size == 0:
        raise ValueError("Cannot compute circular mean from empty array.")

    return float(
        np.arctan2(
            np.mean(np.sin(angles_rad)),
            np.mean(np.cos(angles_rad)),
        )
    )


# ============================================================
# FILE SEARCH
# ============================================================


def find_elastin_mask(mosaic_dir: Path) -> Optional[Path]:
    seg_dir = mosaic_dir / SEGMENTATION_SUBDIR

    if not seg_dir.exists():
        return None

    candidates = sorted(
        list(seg_dir.glob(f"*{ELASTIN_MASK_SUFFIX}"))
        + list(seg_dir.glob("*_elastin_mask.tiff")),
        key=natural_key,
    )

    if len(candidates) == 0:
        return None

    if len(candidates) > 1:
        print(f"[WARN] Multiple elastin masks found in {seg_dir}, using first:")
        for c in candidates:
            print(f"       - {c.name}")

    return candidates[0]


def collect_cases(patient_dir: Path) -> list[Case]:
    cases: list[Case] = []

    visit_dirs = sorted(
        [p for p in patient_dir.glob("visit*") if p.is_dir()],
        key=natural_key,
    )

    for visit_dir in visit_dirs:
        mosaic_dirs = sorted(
            [p for p in visit_dir.glob("Mosaic*") if p.is_dir()],
            key=natural_key,
        )

        for mosaic_dir in mosaic_dirs:
            phasor_dir = mosaic_dir / PHASOR_SUBDIR

            raw_phasor_path = phasor_dir / RAW_PHASOR_NAME
            calibrated_phasor_path = phasor_dir / CALIBRATED_PHASOR_NAME
            elastin_mask_path = find_elastin_mask(mosaic_dir)

            if not raw_phasor_path.exists():
                print(f"[SKIP] Missing raw phasor: {raw_phasor_path}")
                continue

            if not calibrated_phasor_path.exists():
                print(f"[SKIP] Missing calibrated phasor: {calibrated_phasor_path}")
                continue

            if elastin_mask_path is None:
                print(
                    f"[SKIP] Missing elastin mask: "
                    f"{mosaic_dir / SEGMENTATION_SUBDIR}"
                )
                continue

            patient, visit = infer_patient_visit_from_path(mosaic_dir)

            cases.append(
                Case(
                    patient=patient,
                    visit=visit,
                    mosaic_name=mosaic_dir.name,
                    mosaic_dir=mosaic_dir,
                    raw_phasor_path=raw_phasor_path,
                    calibrated_phasor_path=calibrated_phasor_path,
                    elastin_mask_path=elastin_mask_path,
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
            mask = mask[..., :3].max(axis=-1)
        else:
            raise ValueError(f"Unsupported mask shape {mask.shape} in {mask_path}")

    if mask.dtype == bool:
        return label(mask).astype(np.int32)

    unique_vals = np.unique(mask)

    if set(unique_vals.tolist()).issubset({0, 1, 255}):
        return label(mask > 0).astype(np.int32)

    return mask.astype(np.int32)


def read_phasor_stack(phasor_path: Path) -> np.ndarray:
    stack = tifffile.imread(phasor_path)
    stack = np.asarray(stack).squeeze()

    if stack.ndim != 3:
        raise ValueError(
            f"Expected phasor stack CYX, got {stack.shape} in {phasor_path}"
        )

    if stack.shape[0] < 5:
        raise ValueError(
            f"Expected at least 5 planes, got {stack.shape[0]} in {phasor_path}"
        )

    return stack.astype(np.float32, copy=False)


# ============================================================
# ROI EXTRACTION
# ============================================================


def roi_table_from_stack(
    *,
    case: Case,
    stack: np.ndarray,
    labels: np.ndarray,
    phasor_state: str,
    phasor_path: Path,
) -> pd.DataFrame:
    dc = stack[DC_IDX].astype(np.float64)
    g_green = stack[G_GREEN_IDX].astype(np.float64)
    s_green = stack[S_GREEN_IDX].astype(np.float64)
    g_blue = stack[G_BLUE_IDX].astype(np.float64)
    s_blue = stack[S_BLUE_IDX].astype(np.float64)

    if labels.shape != g_green.shape:
        raise ValueError(
            f"Shape mismatch for {case.mosaic_name}: "
            f"phasor={g_green.shape}, mask={labels.shape}"
        )

    rows = []

    for prop in regionprops(labels):
        if prop.area < MIN_ROI_AREA:
            continue

        rr = prop.coords[:, 0]
        cc = prop.coords[:, 1]

        dc_vals = dc[rr, cc]

        gg = g_green[rr, cc]
        sg = s_green[rr, cc]

        gb = g_blue[rr, cc]
        sb = s_blue[rr, cc]

        valid_green = np.isfinite(gg) & np.isfinite(sg)
        valid_blue = np.isfinite(gb) & np.isfinite(sb)

        if FILTER_PHASOR_RANGE:
            valid_green &= (gg >= G_MIN) & (gg <= G_MAX) & (sg >= S_MIN) & (sg <= S_MAX)
            valid_blue &= (gb >= G_MIN) & (gb <= G_MAX) & (sb >= S_MIN) & (sb <= S_MAX)

        if valid_green.sum() < MIN_VALID_PIXELS and valid_blue.sum() < MIN_VALID_PIXELS:
            continue

        row = {
            "patient": case.patient,
            "visit": case.visit,
            "mosaic_name": case.mosaic_name,
            "mosaic_dir": str(case.mosaic_dir),
            "phasor_state": phasor_state,
            "phasor_path": str(phasor_path),
            "elastin_mask_path": str(case.elastin_mask_path),
            "roi_label": int(prop.label),
            "area_px": int(prop.area),
            "centroid_row": float(prop.centroid[0]),
            "centroid_col": float(prop.centroid[1]),
            "dc_mean": float(np.nanmean(dc_vals)),
        }

        if valid_green.sum() >= MIN_VALID_PIXELS:
            gg_valid = gg[valid_green]
            sg_valid = sg[valid_green]

            row.update(
                {
                    "g_green_mean": float(np.mean(gg_valid)),
                    "s_green_mean": float(np.mean(sg_valid)),
                    "g_green_std": float(np.std(gg_valid, ddof=0)),
                    "s_green_std": float(np.std(sg_valid, ddof=0)),
                    "n_valid_green": int(valid_green.sum()),
                }
            )
        else:
            row.update(
                {
                    "g_green_mean": np.nan,
                    "s_green_mean": np.nan,
                    "g_green_std": np.nan,
                    "s_green_std": np.nan,
                    "n_valid_green": 0,
                }
            )

        if valid_blue.sum() >= MIN_VALID_PIXELS:
            gb_valid = gb[valid_blue]
            sb_valid = sb[valid_blue]

            row.update(
                {
                    "g_blue_mean": float(np.mean(gb_valid)),
                    "s_blue_mean": float(np.mean(sb_valid)),
                    "g_blue_std": float(np.std(gb_valid, ddof=0)),
                    "s_blue_std": float(np.std(sb_valid, ddof=0)),
                    "n_valid_blue": int(valid_blue.sum()),
                }
            )
        else:
            row.update(
                {
                    "g_blue_mean": np.nan,
                    "s_blue_mean": np.nan,
                    "g_blue_std": np.nan,
                    "s_blue_std": np.nan,
                    "n_valid_blue": 0,
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


def roi_tables_from_case(case: Case) -> list[pd.DataFrame]:
    labels = read_mask(case.elastin_mask_path)

    raw_stack = read_phasor_stack(case.raw_phasor_path)
    calibrated_stack = read_phasor_stack(case.calibrated_phasor_path)

    df_raw = roi_table_from_stack(
        case=case,
        stack=raw_stack,
        labels=labels,
        phasor_state="raw",
        phasor_path=case.raw_phasor_path,
    )

    df_calibrated = roi_table_from_stack(
        case=case,
        stack=calibrated_stack,
        labels=labels,
        phasor_state="calibrated",
        phasor_path=case.calibrated_phasor_path,
    )

    return [df_raw, df_calibrated]


# ============================================================
# CORRECTION PARAMETERS
# ============================================================


def compute_state_visit_centroids(
    df_roi: pd.DataFrame,
    *,
    channel_name: str,
    g_col: str,
    s_col: str,
) -> pd.DataFrame:
    df = df_roi.copy()
    df = df[np.isfinite(df[g_col]) & np.isfinite(df[s_col])].copy()

    if df.empty:
        raise ValueError(f"No valid ROI rows for channel {channel_name}")

    centroids = (
        df.groupby(["phasor_state", "visit"])
        .agg(
            centroid_g=(g_col, "mean"),
            centroid_s=(s_col, "mean"),
            sd_g=(g_col, "std"),
            sd_s=(s_col, "std"),
            n_reference=(g_col, "size"),
            total_area_px=("area_px", "sum"),
        )
        .reset_index()
    )

    centroids["sd_g"] = centroids["sd_g"].fillna(0.0)
    centroids["sd_s"] = centroids["sd_s"].fillna(0.0)

    mod, phi = phasor_to_polar(
        centroids["centroid_g"].to_numpy(),
        centroids["centroid_s"].to_numpy(),
    )

    centroids["mod"] = mod
    centroids["phi"] = phi
    centroids["channel"] = channel_name

    return centroids


def compute_channel_correction(
    df_roi: pd.DataFrame,
    *,
    channel_name: str,
    g_col: str,
    s_col: str,
) -> pd.DataFrame:
    centroids = compute_state_visit_centroids(
        df_roi,
        channel_name=channel_name,
        g_col=g_col,
        s_col=s_col,
    )

    raw_centroids = centroids[centroids["phasor_state"] == "raw"].copy()
    calibrated_centroids = centroids[centroids["phasor_state"] == "calibrated"].copy()

    if raw_centroids.empty:
        raise ValueError(f"No raw centroids found for channel {channel_name}")

    if calibrated_centroids.empty:
        raise ValueError(f"No calibrated centroids found for channel {channel_name}")

    # Target reference:
    # equal-weight global average of calibrated elastin centroids across visits.
    mod_ref = float(calibrated_centroids["mod"].mean())
    phi_ref = circular_mean(calibrated_centroids["phi"].to_numpy())

    g_ref = float(mod_ref * np.cos(phi_ref))
    s_ref = float(mod_ref * np.sin(phi_ref))

    params = raw_centroids.copy()

    params = params.rename(
        columns={
            "centroid_g": "source_raw_centroid_g",
            "centroid_s": "source_raw_centroid_s",
            "sd_g": "source_raw_sd_g",
            "sd_s": "source_raw_sd_s",
            "mod": "source_raw_mod",
            "phi": "source_raw_phi",
        }
    )

    params["target_state"] = "calibrated_global"
    params["target_mod_ref"] = mod_ref
    params["target_phi_ref"] = phi_ref
    params["target_g_ref"] = g_ref
    params["target_s_ref"] = s_ref

    params["dphi"] = params["target_phi_ref"] - params["source_raw_phi"]
    params["mod_scale"] = params["target_mod_ref"] / params["source_raw_mod"]

    # Expected corrected centroid after applying the correction.
    params["expected_corrected_g"] = (
        params["source_raw_mod"]
        * params["mod_scale"]
        * np.cos(params["source_raw_phi"] + params["dphi"])
    )
    params["expected_corrected_s"] = (
        params["source_raw_mod"]
        * params["mod_scale"]
        * np.sin(params["source_raw_phi"] + params["dphi"])
    )

    params["distance_raw_to_target_before"] = np.sqrt(
        (params["source_raw_centroid_g"] - params["target_g_ref"]) ** 2
        + (params["source_raw_centroid_s"] - params["target_s_ref"]) ** 2
    )

    params["distance_expected_to_target_after"] = np.sqrt(
        (params["expected_corrected_g"] - params["target_g_ref"]) ** 2
        + (params["expected_corrected_s"] - params["target_s_ref"]) ** 2
    )

    params = params[
        [
            "visit",
            "channel",
            "source_raw_centroid_g",
            "source_raw_centroid_s",
            "source_raw_sd_g",
            "source_raw_sd_s",
            "n_reference",
            "total_area_px",
            "source_raw_mod",
            "source_raw_phi",
            "target_state",
            "target_mod_ref",
            "target_phi_ref",
            "target_g_ref",
            "target_s_ref",
            "dphi",
            "mod_scale",
            "expected_corrected_g",
            "expected_corrected_s",
            "distance_raw_to_target_before",
            "distance_expected_to_target_after",
        ]
    ]

    return params


def compute_correction_parameters(df_roi: pd.DataFrame) -> pd.DataFrame:
    green_params = compute_channel_correction(
        df_roi,
        channel_name="green",
        g_col="g_green_mean",
        s_col="s_green_mean",
    )

    blue_params = compute_channel_correction(
        df_roi,
        channel_name="blue",
        g_col="g_blue_mean",
        s_col="s_blue_mean",
    )

    return pd.concat([green_params, blue_params], ignore_index=True)


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    cases = collect_cases(PATIENT_DIR)

    if not cases:
        raise RuntimeError("No valid mosaics found.")

    print(
        f"[INFO] Found {len(cases)} mosaics with raw/calibrated phasors "
        f"+ elastin mask"
    )

    all_tables = []

    for case in cases:
        try:
            dfs = roi_tables_from_case(case)

            for df_case in dfs:
                if len(df_case) == 0:
                    print(
                        f"[WARN] No elastin extracted: "
                        f"{case.visit} | {case.mosaic_name}"
                    )
                    continue

                all_tables.append(df_case)

                state = df_case["phasor_state"].iloc[0]
                print(
                    f"[OK] {case.visit} | {case.mosaic_name} | "
                    f"{state} elastin ROIs = {len(df_case)}"
                )

        except Exception as e:
            print(f"[ERROR] {case.visit} | {case.mosaic_name}: {e}")

    if not all_tables:
        raise RuntimeError("No ROI tables were generated.")

    df_roi = pd.concat(all_tables, ignore_index=True)
    df_roi.to_csv(ROI_OUTPUT_CSV, index=False)

    print("\n[INFO] Saved elastin ROI table:")
    print(f"       {ROI_OUTPUT_CSV}")

    params = compute_correction_parameters(df_roi)
    params.to_csv(PARAMS_OUTPUT_CSV, index=False)

    print("\n[INFO] Saved correction parameters:")
    print(f"       {PARAMS_OUTPUT_CSV}")

    print("\nCorrection summary:")
    print(
        params[
            [
                "visit",
                "channel",
                "source_raw_centroid_g",
                "source_raw_centroid_s",
                "target_g_ref",
                "target_s_ref",
                "dphi",
                "mod_scale",
                "distance_raw_to_target_before",
                "distance_expected_to_target_after",
            ]
        ].to_string(index=False)
    )

    print("\n[DONE]")


if __name__ == "__main__":
    main()
