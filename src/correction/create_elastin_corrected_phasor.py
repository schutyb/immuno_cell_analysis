#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Apply global elastin-based correction to raw phasor mosaics.

Input:
    phasor/phasor_raw_green_blue_mosaic.tif

Correction parameters:
    analysis/elastin_correction/elastin_correction_parameters_raw_global.csv

Output:
    phasor/phasor_raw_green_blue_mosaic_elastin_corrected.tif

Input/output planes:
    0 = DC
    1 = G green
    2 = S green
    3 = G blue
    4 = S blue
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

# ============================================================
# CONFIG
# ============================================================

PATIENT_DIR = Path(
    "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449"
).expanduser()

PHASOR_SUBDIR = "phasor"

RAW_PHASOR_NAME = "phasor_raw_green_blue_mosaic.tif"
CORRECTED_PHASOR_NAME = "phasor_raw_green_blue_mosaic_elastin_corrected.tif"
CORRECTED_METADATA_NAME = "phasor_raw_green_blue_mosaic_elastin_corrected_metadata.txt"

PARAMS_CSV = (
    PATIENT_DIR
    / "analysis"
    / "elastin_correction"
    / "elastin_correction_parameters_raw_global.csv"
)

OVERWRITE = True

# TIFF planes
DC_IDX = 0
G_GREEN_IDX = 1
S_GREEN_IDX = 2
G_BLUE_IDX = 3
S_BLUE_IDX = 4

# Apply correction only where DC is above threshold.
# Use None to correct all finite pixels.
VALID_DC_THRESHOLD = 0.0


# ============================================================
# HELPERS
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


def polar_to_phasor(mod: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mod = np.asarray(mod, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)

    g = mod * np.cos(phi)
    s = mod * np.sin(phi)

    return g, s


def get_params_for_visit_channel(
    params: pd.DataFrame,
    visit: str,
    channel: str,
) -> tuple[float, float]:
    rows = params[
        (params["visit"].astype(str) == str(visit))
        & (params["channel"].astype(str) == str(channel))
    ]

    if rows.empty:
        raise KeyError(
            f"No correction parameters found for visit={visit}, channel={channel}"
        )

    if len(rows) > 1:
        raise ValueError(
            f"Multiple parameter rows found for visit={visit}, channel={channel}"
        )

    dphi = float(rows["dphi"].iloc[0])
    mod_scale = float(rows["mod_scale"].iloc[0])

    return dphi, mod_scale


def correct_channel(
    g: np.ndarray,
    s: np.ndarray,
    dc: np.ndarray,
    *,
    dphi: float,
    mod_scale: float,
    valid_dc_threshold: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(g, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    dc = np.asarray(dc, dtype=np.float64)

    valid = np.isfinite(g) & np.isfinite(s)

    if valid_dc_threshold is not None:
        valid &= np.isfinite(dc) & (dc > valid_dc_threshold)

    mod, phi = phasor_to_polar(g, s)

    mod_corr = mod.copy()
    phi_corr = phi.copy()

    mod_corr[valid] = mod[valid] * mod_scale
    phi_corr[valid] = phi[valid] + dphi

    g_corr, s_corr = polar_to_phasor(mod_corr, phi_corr)

    g_corr = g_corr.astype(np.float32)
    s_corr = s_corr.astype(np.float32)

    g_corr[~np.isfinite(g_corr)] = np.nan
    s_corr[~np.isfinite(s_corr)] = np.nan

    return g_corr, s_corr


def read_raw_phasor(path: Path) -> np.ndarray:
    stack = tifffile.imread(path)
    stack = np.asarray(stack).squeeze()

    if stack.ndim != 3:
        raise ValueError(f"Expected CYX phasor stack, got {stack.shape} in {path}")

    if stack.shape[0] < 5:
        raise ValueError(f"Expected at least 5 planes, got {stack.shape[0]} in {path}")

    return stack.astype(np.float32, copy=False)


def write_metadata(
    metadata_path: Path,
    *,
    raw_path: Path,
    corrected_path: Path,
    visit: str,
    green_dphi: float,
    green_mod_scale: float,
    blue_dphi: float,
    blue_mod_scale: float,
    params_csv: Path,
    corrected_stack: np.ndarray,
) -> None:
    text = f"""Elastin-corrected raw phasor metadata

Input raw phasor:
{raw_path}

Output corrected phasor:
{corrected_path}

Correction parameters CSV:
{params_csv}

Visit:
{visit}

Output dtype:
float32

Output shape:
{corrected_stack.shape}

Axis order:
plane, y, x

Planes:
0 = DC intensity image
1 = elastin-corrected raw G / real component, green detector
2 = elastin-corrected raw S / imaginary component, green detector
3 = elastin-corrected raw G / real component, blue detector
4 = elastin-corrected raw S / imaginary component, blue detector

Correction model:
The correction is applied in polar phasor coordinates independently for each detector
channel.

For each pixel:
    modulation_corrected = modulation_raw * mod_scale
    phase_corrected = phase_raw + dphi

Green channel correction:
dphi = {green_dphi}
mod_scale = {green_mod_scale}

Blue channel correction:
dphi = {blue_dphi}
mod_scale = {blue_mod_scale}

Valid DC threshold:
{VALID_DC_THRESHOLD}

Invalid pixels:
Invalid G/S pixels are stored as NaN.
DC is copied from the raw phasor TIFF.
"""
    metadata_path.write_text(text)


# ============================================================
# PROCESSING
# ============================================================


def correct_one_mosaic(raw_path: Path, params: pd.DataFrame) -> None:
    mosaic_dir = raw_path.parent.parent
    _, visit = infer_patient_visit_from_path(mosaic_dir)

    out_path = raw_path.parent / CORRECTED_PHASOR_NAME
    metadata_path = raw_path.parent / CORRECTED_METADATA_NAME

    if out_path.exists() and not OVERWRITE:
        print(f"[SKIP] Exists: {out_path}")
        return

    green_dphi, green_mod_scale = get_params_for_visit_channel(
        params,
        visit=visit,
        channel="green",
    )

    blue_dphi, blue_mod_scale = get_params_for_visit_channel(
        params,
        visit=visit,
        channel="blue",
    )

    raw = read_raw_phasor(raw_path)

    corrected = raw.astype(np.float32, copy=True)

    dc = raw[DC_IDX]

    g_green_corr, s_green_corr = correct_channel(
        raw[G_GREEN_IDX],
        raw[S_GREEN_IDX],
        dc,
        dphi=green_dphi,
        mod_scale=green_mod_scale,
        valid_dc_threshold=VALID_DC_THRESHOLD,
    )

    g_blue_corr, s_blue_corr = correct_channel(
        raw[G_BLUE_IDX],
        raw[S_BLUE_IDX],
        dc,
        dphi=blue_dphi,
        mod_scale=blue_mod_scale,
        valid_dc_threshold=VALID_DC_THRESHOLD,
    )

    corrected[DC_IDX] = dc.astype(np.float32)
    corrected[G_GREEN_IDX] = g_green_corr
    corrected[S_GREEN_IDX] = s_green_corr
    corrected[G_BLUE_IDX] = g_blue_corr
    corrected[S_BLUE_IDX] = s_blue_corr

    tifffile.imwrite(
        out_path,
        corrected.astype(np.float32),
        imagej=False,
        metadata={
            "axes": "CYX",
            "planes": (
                "0=DC, "
                "1=G_green_elastin_corrected, "
                "2=S_green_elastin_corrected, "
                "3=G_blue_elastin_corrected, "
                "4=S_blue_elastin_corrected"
            ),
        },
    )

    write_metadata(
        metadata_path,
        raw_path=raw_path,
        corrected_path=out_path,
        visit=visit,
        green_dphi=green_dphi,
        green_mod_scale=green_mod_scale,
        blue_dphi=blue_dphi,
        blue_mod_scale=blue_mod_scale,
        params_csv=PARAMS_CSV,
        corrected_stack=corrected,
    )

    print(f"[OK] {visit} | {mosaic_dir.name}")
    print(f"     OUT : {out_path}")
    print(f"     META: {metadata_path}")


def main() -> None:
    if not PARAMS_CSV.exists():
        raise FileNotFoundError(f"Missing correction parameters CSV: {PARAMS_CSV}")

    params = pd.read_csv(PARAMS_CSV)

    required_cols = {"visit", "channel", "dphi", "mod_scale"}
    missing = required_cols - set(params.columns)

    if missing:
        raise ValueError(f"Missing columns in correction CSV: {sorted(missing)}")

    raw_paths = sorted(
        PATIENT_DIR.glob(f"visit*/Mosaic*/{PHASOR_SUBDIR}/{RAW_PHASOR_NAME}"),
        key=natural_key,
    )

    if not raw_paths:
        raise RuntimeError(
            f"No raw phasor files found with pattern: "
            f"{PATIENT_DIR}/visit*/Mosaic*/{PHASOR_SUBDIR}/{RAW_PHASOR_NAME}"
        )

    print(f"[INFO] Found {len(raw_paths)} raw phasor mosaics")
    print(f"[INFO] Parameters: {PARAMS_CSV}")

    for raw_path in raw_paths:
        try:
            correct_one_mosaic(raw_path, params)
        except Exception as e:
            print(f"[ERROR] {raw_path}: {e}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
