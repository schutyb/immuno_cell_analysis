#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create raw and coumarin-calibrated phasor mosaics from raw FLIM tiles.

This script processes all visits and mosaics inside a patient folder. For each
mosaic, it reads the raw FLIM tile stacks, separates the green and blue detector
channels, computes the first-harmonic phasor coordinates, saves the raw phasor,
then calibrates the phasor using the visit-specific coumarin reference and saves
the calibrated phasor.

Outputs are stored inside each mosaic folder under:

    phasor/

Output RAW TIFF planes:
    0 = DC intensity image
    1 = raw G / real component, green detector
    2 = raw S / imaginary component, green detector
    3 = raw G / real component, blue detector
    4 = raw S / imaginary component, blue detector

Output CALIBRATED TIFF planes:
    0 = DC intensity image
    1 = calibrated G / real component, green detector
    2 = calibrated S / imaginary component, green detector
    3 = calibrated G / real component, blue detector
    4 = calibrated S / imaginary component, blue detector
"""

import re
from pathlib import Path

import numpy as np
import tifffile as tiff
from phasorpy.lifetime import phasor_calibrate
from phasorpy.phasor import phasor_from_signal


# =========================
# CONFIG
# =========================

PATIENT_DIR = Path(
    "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449"
).expanduser()

FLIM_SUBDIR = "flim"
COUMARIN_PREFIX = "coumarin"

OUTPUT_SUBDIR = "phasor"

RAW_OUTPUT_NAME = "phasor_raw_green_blue_mosaic.tif"
CAL_OUTPUT_NAME = "phasor_calibrated_green_blue_mosaic.tif"

RAW_METADATA_NAME = "phasor_raw_green_blue_mosaic_metadata.txt"
CAL_METADATA_NAME = "phasor_calibrated_green_blue_mosaic_metadata.txt"

N_GREEN = 16

FREQUENCY_MHZ = 80.0
COUMARIN_LIFETIME_NS = 2.5

OVERWRITE = True


# =========================
# HELPERS
# =========================


def natural_key(path):
    return [
        int(t) if t.isdigit() else t.lower()
        for t in re.split(r"(\d+)", str(path.name))
    ]


def parse_mosaic_shape_from_name(folder_name):
    m = re.search(r"(\d+)x(\d+)", folder_name)
    if not m:
        raise ValueError(f"No pude detectar forma del mosaico en: {folder_name}")

    return int(m.group(1)), int(m.group(2))


def collect_tile_paths(flim_dir):
    files = []

    for p in flim_dir.glob("Im_*.tif"):
        m = re.search(r"Im_(\d+)\.tif$", p.name, re.IGNORECASE)
        if m:
            files.append((int(m.group(1)), p))

    files.sort(key=lambda x: x[0])
    return [p for _, p in files]


def find_coumarin_file(visit_dir):
    coumarin_dirs = [
        p
        for p in visit_dir.iterdir()
        if p.is_dir() and p.name.lower().startswith(COUMARIN_PREFIX)
    ]

    if len(coumarin_dirs) == 0:
        raise FileNotFoundError(f"No encontré carpeta coumarin en {visit_dir}")

    coumarin_dir = sorted(coumarin_dirs, key=natural_key)[0]

    tif_files = sorted(
        list(coumarin_dir.rglob("*.tif")) + list(coumarin_dir.rglob("*.tiff")),
        key=natural_key,
    )

    tif_files = [p for p in tif_files if p.name.lower().startswith("im_")]

    if len(tif_files) == 0:
        raise FileNotFoundError(f"No encontré TIFF de coumarin en {coumarin_dir}")

    return tif_files[0]


def read_stack_tyx(path):
    """
    Read FLIM TIFF and return stack as:

        T, Y, X

    T is usually 31 or 32.
    """
    arr = tiff.imread(path)
    arr = np.asarray(arr)

    if arr.ndim != 3:
        raise ValueError(f"Esperaba stack 3D, recibí {arr.shape} en {path}")

    time_axis = int(np.argmin(arr.shape))

    if time_axis != 0:
        arr = np.moveaxis(arr, time_axis, 0)

    return arr.astype(np.float32, copy=False)


def split_green_blue(stack_tyx):
    """
    First 16 bins: green detector.
    Remaining bins: blue detector.
    """
    nt = stack_tyx.shape[0]

    if nt <= N_GREEN:
        raise ValueError(
            f"El stack tiene {nt} bins, no alcanza para separar green={N_GREEN} + blue"
        )

    green = stack_tyx[:N_GREEN]
    blue = stack_tyx[N_GREEN:]

    return green, blue


def compute_phasor_from_decay(decay_tyx):
    """
    decay_tyx:
        T, Y, X

    PhasorPy with numpy ndarray uses axis=-1.
    Therefore:
        T, Y, X -> Y, X, T
    """
    signal_yxt = np.moveaxis(decay_tyx, 0, -1).astype(np.float32, copy=False)

    mean, real, imag = phasor_from_signal(signal_yxt, axis=-1)

    mean = np.asarray(mean, dtype=np.float32)
    real = np.asarray(real, dtype=np.float32)
    imag = np.asarray(imag, dtype=np.float32)

    mean[~np.isfinite(mean)] = np.nan
    real[~np.isfinite(real)] = np.nan
    imag[~np.isfinite(imag)] = np.nan

    return mean, real, imag


def compute_coumarin_reference(coumarin_path):
    stack = read_stack_tyx(coumarin_path)
    green, blue = split_green_blue(stack)

    mean_g, real_g, imag_g = compute_phasor_from_decay(green)
    mean_b, real_b, imag_b = compute_phasor_from_decay(blue)

    return {
        "path": coumarin_path,
        "green": {
            "mean": float(np.nanmean(mean_g)),
            "real": float(np.nanmean(real_g)),
            "imag": float(np.nanmean(imag_g)),
        },
        "blue": {
            "mean": float(np.nanmean(mean_b)),
            "real": float(np.nanmean(real_b)),
            "imag": float(np.nanmean(imag_b)),
        },
    }


def calibrate_phasor(real, imag, ref):
    real_cal, imag_cal = phasor_calibrate(
        real,
        imag,
        ref["mean"],
        ref["real"],
        ref["imag"],
        frequency=FREQUENCY_MHZ,
        lifetime=COUMARIN_LIFETIME_NS,
    )

    real_cal = np.asarray(real_cal, dtype=np.float32)
    imag_cal = np.asarray(imag_cal, dtype=np.float32)

    real_cal[~np.isfinite(real_cal)] = np.nan
    imag_cal[~np.isfinite(imag_cal)] = np.nan

    return real_cal, imag_cal


def snake_indices(nrows, ncols):
    idx = np.arange(nrows * ncols).reshape(nrows, ncols)
    layout = []

    for r in range(nrows):
        row = idx[r].copy()

        if r % 2 == 1:
            row = row[::-1]

        layout.append(row.tolist())

    return layout


def stitch_tiles_snake(tile_imgs, nrows, ncols):
    expected = nrows * ncols

    if len(tile_imgs) != expected:
        raise ValueError(f"Esperaba {expected} tiles, recibí {len(tile_imgs)}")

    h, w = tile_imgs[0].shape

    mosaic = np.full(
        (nrows * h, ncols * w),
        np.nan,
        dtype=np.float32,
    )

    layout = snake_indices(nrows, ncols)

    for r in range(nrows):
        for c in range(ncols):
            tile_idx = layout[r][c]

            y0 = r * h
            y1 = y0 + h
            x0 = c * w
            x1 = x0 + w

            mosaic[y0:y1, x0:x1] = tile_imgs[tile_idx]

    return mosaic


def write_metadata(
    metadata_path,
    title,
    mosaic_dir,
    tile_paths,
    coumarin_ref,
    output_path,
    phasor_stack,
    is_calibrated,
):
    if is_calibrated:
        planes = """0 = DC intensity image
1 = calibrated G / real component, green detector
2 = calibrated S / imaginary component, green detector
3 = calibrated G / real component, blue detector
4 = calibrated S / imaginary component, blue detector"""
        calibration_text = f"""Calibration:
reference = coumarin
coumarin lifetime = {COUMARIN_LIFETIME_NS} ns
laser frequency = {FREQUENCY_MHZ} MHz"""
    else:
        planes = """0 = DC intensity image
1 = raw G / real component, green detector
2 = raw S / imaginary component, green detector
3 = raw G / real component, blue detector
4 = raw S / imaginary component, blue detector"""
        calibration_text = """Calibration:
none
This TIFF stores the raw first-harmonic phasor before coumarin calibration."""

    text = f"""{title}

Patient directory:
{PATIENT_DIR}

Mosaic directory:
{mosaic_dir}

Output TIFF:
{output_path}

Output dtype:
float32

Output shape:
{phasor_stack.shape}

Axis order:
plane, y, x

Planes:
{planes}

Detector split:
green detector = first {N_GREEN} bins
blue detector = remaining bins

Invalid phasor pixels:
G/S invalid pixels are stored as NaN, not zero.
Use np.nanmean() for ROI-level averaging.

{calibration_text}

Coumarin file:
{coumarin_ref["path"]}

Coumarin green reference:
mean = {coumarin_ref["green"]["mean"]}
real = {coumarin_ref["green"]["real"]}
imag = {coumarin_ref["green"]["imag"]}

Coumarin blue reference:
mean = {coumarin_ref["blue"]["mean"]}
real = {coumarin_ref["blue"]["real"]}
imag = {coumarin_ref["blue"]["imag"]}

Tiles used:
"""

    for p in tile_paths:
        text += f"- {p.name}\n"

    metadata_path.write_text(text)


# =========================
# PROCESSING
# =========================


def process_mosaic(mosaic_dir, coumarin_ref):
    flim_dir = mosaic_dir / FLIM_SUBDIR
    out_dir = mosaic_dir / OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_out_path = out_dir / RAW_OUTPUT_NAME
    cal_out_path = out_dir / CAL_OUTPUT_NAME

    raw_metadata_path = out_dir / RAW_METADATA_NAME
    cal_metadata_path = out_dir / CAL_METADATA_NAME

    if (
        raw_out_path.exists()
        and cal_out_path.exists()
        and not OVERWRITE
    ):
        print(f"[SKIP] Existen RAW y CAL: {mosaic_dir.name}")
        return

    if not flim_dir.exists():
        print(f"[SKIP] No existe flim/: {mosaic_dir}")
        return

    nrows, ncols = parse_mosaic_shape_from_name(mosaic_dir.name)
    tile_paths = collect_tile_paths(flim_dir)

    expected = nrows * ncols

    if len(tile_paths) != expected:
        raise ValueError(
            f"{mosaic_dir.name}: esperaba {expected} tiles, encontré {len(tile_paths)}"
        )

    dc_tiles = []

    g_green_raw_tiles = []
    s_green_raw_tiles = []
    g_blue_raw_tiles = []
    s_blue_raw_tiles = []

    g_green_cal_tiles = []
    s_green_cal_tiles = []
    g_blue_cal_tiles = []
    s_blue_cal_tiles = []

    for tile_path in tile_paths:
        stack = read_stack_tyx(tile_path)
        green, blue = split_green_blue(stack)

        _, real_green, imag_green = compute_phasor_from_decay(green)
        _, real_blue, imag_blue = compute_phasor_from_decay(blue)

        # -------------------------
        # RAW PHASOR
        # -------------------------
        g_green_raw_tiles.append(real_green.astype(np.float32))
        s_green_raw_tiles.append(imag_green.astype(np.float32))
        g_blue_raw_tiles.append(real_blue.astype(np.float32))
        s_blue_raw_tiles.append(imag_blue.astype(np.float32))

        # -------------------------
        # COUMARIN-CALIBRATED PHASOR
        # -------------------------
        real_green_cal, imag_green_cal = calibrate_phasor(
            real_green,
            imag_green,
            coumarin_ref["green"],
        )

        real_blue_cal, imag_blue_cal = calibrate_phasor(
            real_blue,
            imag_blue,
            coumarin_ref["blue"],
        )

        g_green_cal_tiles.append(real_green_cal)
        s_green_cal_tiles.append(imag_green_cal)
        g_blue_cal_tiles.append(real_blue_cal)
        s_blue_cal_tiles.append(imag_blue_cal)

        # DC total, not phasorpy mean.
        dc = green.sum(axis=0).astype(np.float32) + blue.sum(axis=0).astype(np.float32)
        dc[~np.isfinite(dc)] = 0
        dc_tiles.append(dc)

    dc_mosaic = stitch_tiles_snake(dc_tiles, nrows, ncols)
    dc_mosaic[~np.isfinite(dc_mosaic)] = 0

    # -------------------------
    # RAW MOSAICS
    # -------------------------
    g_green_raw_mosaic = stitch_tiles_snake(g_green_raw_tiles, nrows, ncols)
    s_green_raw_mosaic = stitch_tiles_snake(s_green_raw_tiles, nrows, ncols)
    g_blue_raw_mosaic = stitch_tiles_snake(g_blue_raw_tiles, nrows, ncols)
    s_blue_raw_mosaic = stitch_tiles_snake(s_blue_raw_tiles, nrows, ncols)

    raw_phasor_stack = np.stack(
        [
            dc_mosaic,
            g_green_raw_mosaic,
            s_green_raw_mosaic,
            g_blue_raw_mosaic,
            s_blue_raw_mosaic,
        ],
        axis=0,
    ).astype(np.float32)

    # -------------------------
    # CALIBRATED MOSAICS
    # -------------------------
    g_green_cal_mosaic = stitch_tiles_snake(g_green_cal_tiles, nrows, ncols)
    s_green_cal_mosaic = stitch_tiles_snake(s_green_cal_tiles, nrows, ncols)
    g_blue_cal_mosaic = stitch_tiles_snake(g_blue_cal_tiles, nrows, ncols)
    s_blue_cal_mosaic = stitch_tiles_snake(s_blue_cal_tiles, nrows, ncols)

    cal_phasor_stack = np.stack(
        [
            dc_mosaic,
            g_green_cal_mosaic,
            s_green_cal_mosaic,
            g_blue_cal_mosaic,
            s_blue_cal_mosaic,
        ],
        axis=0,
    ).astype(np.float32)

    # -------------------------
    # SAVE RAW
    # -------------------------
    if OVERWRITE or not raw_out_path.exists():
        tiff.imwrite(
            raw_out_path,
            raw_phasor_stack,
            dtype=np.float32,
            imagej=False,
            metadata={
                "axes": "CYX",
                "planes": (
                    "0=DC, "
                    "1=G_green_raw, 2=S_green_raw, "
                    "3=G_blue_raw, 4=S_blue_raw"
                ),
            },
        )

        write_metadata(
            metadata_path=raw_metadata_path,
            title="Raw phasor mosaic metadata",
            mosaic_dir=mosaic_dir,
            tile_paths=tile_paths,
            coumarin_ref=coumarin_ref,
            output_path=raw_out_path,
            phasor_stack=raw_phasor_stack,
            is_calibrated=False,
        )

    # -------------------------
    # SAVE CALIBRATED
    # -------------------------
    if OVERWRITE or not cal_out_path.exists():
        tiff.imwrite(
            cal_out_path,
            cal_phasor_stack,
            dtype=np.float32,
            imagej=False,
            metadata={
                "axes": "CYX",
                "planes": (
                    "0=DC, "
                    "1=G_green_cal, 2=S_green_cal, "
                    "3=G_blue_cal, 4=S_blue_cal"
                ),
            },
        )

        write_metadata(
            metadata_path=cal_metadata_path,
            title="Coumarin-calibrated phasor mosaic metadata",
            mosaic_dir=mosaic_dir,
            tile_paths=tile_paths,
            coumarin_ref=coumarin_ref,
            output_path=cal_out_path,
            phasor_stack=cal_phasor_stack,
            is_calibrated=True,
        )

    print(f"[OK] {mosaic_dir.name}")
    print(f"     RAW TIFF: {raw_out_path}")
    print(f"     RAW META: {raw_metadata_path}")
    print(f"     CAL TIFF: {cal_out_path}")
    print(f"     CAL META: {cal_metadata_path}")


def main():
    visit_dirs = sorted(
        [p for p in PATIENT_DIR.glob("visit*") if p.is_dir()],
        key=natural_key,
    )

    for visit_dir in visit_dirs:
        print("\n==============================")
        print(f"Visit: {visit_dir.name}")
        print("==============================")

        coumarin_path = find_coumarin_file(visit_dir)
        coumarin_ref = compute_coumarin_reference(coumarin_path)

        print(f"[INFO] Coumarin: {coumarin_path}")
        print(f"[INFO] Green ref: {coumarin_ref['green']}")
        print(f"[INFO] Blue ref: {coumarin_ref['blue']}")

        mosaic_dirs = sorted(
            [p for p in visit_dir.glob("Mosaic*") if p.is_dir()],
            key=natural_key,
        )

        for mosaic_dir in mosaic_dirs:
            process_mosaic(mosaic_dir, coumarin_ref)

    print("\nListo.")


if __name__ == "__main__":
    main()