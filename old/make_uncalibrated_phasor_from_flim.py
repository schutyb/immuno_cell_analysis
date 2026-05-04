#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tifffile


# =========================
# CONFIG
# =========================

PATIENT_DIR = Path("/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449")

OUTPUT_NAME = "phasor_uncalibrated.tif"

# if True, overwrite existing output
OVERWRITE = True

# supported tile filename pattern
TILE_REGEX = re.compile(r"Im_(\d+)\.tif$", re.IGNORECASE)


# =========================
# HELPERS
# =========================

def parse_mosaic_shape_from_name(folder_name: str) -> Tuple[int, int]:
    """
    Parse mosaic shape from folder name, e.g.:
        Mosaico3_4x4_FOV600_z110_32Sp  -> (4, 4)
        Mosaic03_2x2_FOV...            -> (2, 2)
    """
    m = re.search(r"(\d+)x(\d+)", folder_name)
    if not m:
        raise ValueError(f"Could not parse mosaic shape from folder name: {folder_name}")
    nrows = int(m.group(1))
    ncols = int(m.group(2))
    return nrows, ncols


def snake_indices(nrows: int, ncols: int) -> List[List[int]]:
    """
    Return tile indices (0-based) arranged as snake mosaic.
    Example 4x4:
      [[0,1,2,3],
       [7,6,5,4],
       [8,9,10,11],
       [15,14,13,12]]
    """
    idx = np.arange(nrows * ncols).reshape(nrows, ncols)
    out = []
    for r in range(nrows):
        row = idx[r].copy()
        if r % 2 == 1:
            row = row[::-1]
        out.append(row.tolist())
    return out


def find_flim_dirs(patient_dir: Path) -> List[Path]:
    """
    Find all flim directories under a patient.
    """
    flim_dirs = []
    for p in patient_dir.rglob("flim"):
        if p.is_dir():
            flim_dirs.append(p)
    return sorted(flim_dirs)


def collect_tile_paths(flim_dir: Path) -> List[Path]:
    """
    Collect and sort Im_*.tif tiles by numeric index.
    """
    pairs = []
    for p in flim_dir.iterdir():
        if not p.is_file():
            continue
        m = TILE_REGEX.match(p.name)
        if m:
            pairs.append((int(m.group(1)), p))

    pairs.sort(key=lambda x: x[0])
    return [p for _, p in pairs]


def read_decay_stack(tile_path: Path) -> np.ndarray:
    """
    Read one FLIM tile and return as float64 array.

    Expected typical shape:
      (T, Y, X)

    If data comes as (Y, X, T), it is transposed to (T, Y, X).
    """
    arr = tifffile.imread(tile_path)
    arr = np.asarray(arr)

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D decay stack, got shape {arr.shape} in {tile_path}")

    # Heuristic: time axis is usually the smallest dimension (e.g. 16, 20, 32)
    # while Y,X are much larger (e.g. 256, 512, 1200).
    shape = arr.shape
    time_axis = int(np.argmin(shape))

    if time_axis == 0:
        out = arr
    elif time_axis == 2:
        out = np.moveaxis(arr, 2, 0)
    elif time_axis == 1:
        out = np.moveaxis(arr, 1, 0)
    else:
        out = arr

    return out.astype(np.float64, copy=False)


def select_decay_channels(stack_tyx: np.ndarray, mosaic_name: str) -> np.ndarray:
    """
    Select the decay bins to use for first-harmonic phasor.

    Rules:
    - if folder name contains '32Sp', use first 16 bins
    - otherwise use full decay
    """
    nt = stack_tyx.shape[0]

    if "32sp" in mosaic_name.lower():
        if nt < 16:
            raise ValueError(
                f"Folder suggests 32Sp but stack has only {nt} time bins: {mosaic_name}"
            )
        return stack_tyx[:16]

    return stack_tyx


def compute_first_harmonic_phasor(decay_tyx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute intensity, g, s from decay stack (T, Y, X) using the first harmonic.

    Uncalibrated phasor:
        intensity = sum_t I(t)
        g = sum_t I(t) cos(2π t / T) / sum_t I(t)
        s = sum_t I(t) sin(2π t / T) / sum_t I(t)
    """
    if decay_tyx.ndim != 3:
        raise ValueError(f"Expected (T,Y,X), got {decay_tyx.shape}")

    nt = decay_tyx.shape[0]
    t = np.arange(nt, dtype=np.float64)

    cos_w = np.cos(2.0 * np.pi * t / nt)[:, None, None]
    sin_w = np.sin(2.0 * np.pi * t / nt)[:, None, None]

    intensity = decay_tyx.sum(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        g = (decay_tyx * cos_w).sum(axis=0) / intensity
        s = (decay_tyx * sin_w).sum(axis=0) / intensity

    g = np.where(np.isfinite(g), g, 0.0)
    s = np.where(np.isfinite(s), s, 0.0)
    intensity = np.where(np.isfinite(intensity), intensity, 0.0)

    return intensity.astype(np.float32), g.astype(np.float32), s.astype(np.float32)


def stitch_tiles_snake(
    tile_imgs: List[np.ndarray],
    nrows: int,
    ncols: int,
) -> np.ndarray:
    """
    Stitch a list of 2D tile images into one mosaic using snake ordering.
    """
    expected = nrows * ncols
    if len(tile_imgs) != expected:
        raise ValueError(f"Expected {expected} tiles, got {len(tile_imgs)}")

    h, w = tile_imgs[0].shape
    for i, img in enumerate(tile_imgs):
        if img.shape != (h, w):
            raise ValueError(f"Tile {i} shape mismatch: {img.shape} != {(h, w)}")

    mosaic = np.zeros((nrows * h, ncols * w), dtype=tile_imgs[0].dtype)
    layout = snake_indices(nrows, ncols)

    for r in range(nrows):
        for c in range(ncols):
            tile_idx = layout[r][c]
            y0 = r * h
            y1 = (r + 1) * h
            x0 = c * w
            x1 = (c + 1) * w
            mosaic[y0:y1, x0:x1] = tile_imgs[tile_idx]

    return mosaic


def process_flim_dir(flim_dir: Path) -> Path:
    """
    Process one flim directory and save phasor_uncalibrated.tif
    in the parent mosaic folder.
    """
    mosaic_dir = flim_dir.parent
    mosaic_name = mosaic_dir.name

    nrows, ncols = parse_mosaic_shape_from_name(mosaic_name)
    tile_paths = collect_tile_paths(flim_dir)

    expected = nrows * ncols
    if len(tile_paths) != expected:
        raise ValueError(
            f"{mosaic_dir}: expected {expected} tiles from {nrows}x{ncols}, got {len(tile_paths)}"
        )

    intensity_tiles = []
    g_tiles = []
    s_tiles = []

    for tile_path in tile_paths:
        stack = read_decay_stack(tile_path)
        stack = select_decay_channels(stack, mosaic_name)
        intensity, g, s = compute_first_harmonic_phasor(stack)

        intensity_tiles.append(intensity)
        g_tiles.append(g)
        s_tiles.append(s)

    intensity_mosaic = stitch_tiles_snake(intensity_tiles, nrows, ncols)
    g_mosaic = stitch_tiles_snake(g_tiles, nrows, ncols)
    s_mosaic = stitch_tiles_snake(s_tiles, nrows, ncols)

    out = np.stack([intensity_mosaic, g_mosaic, s_mosaic], axis=0).astype(np.float32)

    out_path = mosaic_dir / OUTPUT_NAME
    if out_path.exists() and not OVERWRITE:
        print(f"[SKIP] Exists: {out_path}")
        return out_path

    tifffile.imwrite(out_path, out)
    return out_path


# =========================
# MAIN
# =========================

def main() -> None:
    flim_dirs = find_flim_dirs(PATIENT_DIR)

    if not flim_dirs:
        raise RuntimeError(f"No flim directories found under: {PATIENT_DIR}")

    print(f"[INFO] Found {len(flim_dirs)} flim folders under {PATIENT_DIR}")

    ok = 0
    fail = 0

    for flim_dir in flim_dirs:
        try:
            out_path = process_flim_dir(flim_dir)
            print(f"[OK] {flim_dir} -> {out_path}")
            ok += 1
        except Exception as e:
            print(f"[ERROR] {flim_dir}")
            print(f"        {type(e).__name__}: {e}")
            fail += 1

    print(f"[DONE] ok={ok} fail={fail}")


if __name__ == "__main__":
    main()