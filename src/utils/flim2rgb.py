#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create pseudo-RGB images from raw FLIM tile stacks.

This script processes all visits and mosaics inside a patient folder. For each
mosaic, it reads the raw FLIM tiles, converts each FLIM decay stack into a
pseudo-RGB image, saves the RGB tiles, and reconstructs a full RGB mosaic using
the snake acquisition layout.

The RGB mapping is based on the detector split used in the DOD immuno-cell
dataset:

    R = sum of the first 2 bins of the green detector
    G = sum of the remaining green detector bins
    B = sum of all blue detector bins

For 32Sp data:
    green detector = first 16 bins
    blue detector = remaining bins

Normalization:
    The same percentile-based normalization is applied across all valid tiles
    within a mosaic. This keeps the RGB scale consistent across tiles from the
    same mosaic.

How to use:
    1. Edit PATIENT_FOLDER to point to the patient directory.
    2. Check N_GREEN and N_TOTAL depending on the acquisition format.
    3. Run from the repository root:

        python -m src.utils.flim2rgb

    or directly:

        python src/utils/flim2rgb.py

Outputs:
    For each mosaic, this script creates:

        <mosaic_dir>/RGB/
            Im_00001_RGB.png
            Im_00002_RGB.png
            ...
            <mosaic_name>_RGB_mosaic.png
            rgb_metadata.txt

Notes:
    - This script does not compute phasors.
    - This script does not perform segmentation.
    - This script is only for visualization and QC of FLIM-derived RGB images.
"""

import re
from pathlib import Path

import numpy as np
import tifffile as tiff
from PIL import Image

# =========================
# CONFIG
# =========================

PATIENT_FOLDER = Path(
    "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p427"
).expanduser()

FLIM_SUBDIR = "flim"
RGB_OUTPUT_SUBDIR = "RGB"

# For 32Sp data:
# first 16 bins = green detector
# remaining bins = blue detector
N_GREEN = 16
N_TOTAL = 31

# Channel scaling before percentile normalization
SCALE_R = 4 / 3
SCALE_G = 3 / 3
SCALE_B = 2 / 3

PERCENTILE_LOW = 1
PERCENTILE_HIGH = 99

PNG_DPI = 600

OVERWRITE = True


# =========================
# HELPERS
# =========================


def natural_key(path):
    """
    Natural sorting key for paths containing numbers.

    Example:
        Im_2.tif comes before Im_10.tif.
    """
    return [
        int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(path.name))
    ]


def parse_mosaic_shape_from_name(folder_name):
    """
    Extract mosaic shape from folder name.

    Example:
        Mosaic03_4x4_FOV600_z110_32Sp -> rows=4, cols=4
    """
    match = re.search(r"(\d+)x(\d+)", folder_name)

    if match is None:
        raise ValueError(
            f"No pude detectar el tamaño del mosaico en: {folder_name}. "
            "Esperaba algo como '4x4' en el nombre."
        )

    rows = int(match.group(1))
    cols = int(match.group(2))

    return rows, cols


def collect_flim_tile_paths(flim_dir):
    """
    Collect raw FLIM tile files named Im_*.tif or Im_*.tiff.
    """
    files = []

    for path in list(flim_dir.glob("Im_*.tif")) + list(flim_dir.glob("Im_*.tiff")):
        match = re.search(r"Im_(\d+)\.tiff?$", path.name, re.IGNORECASE)

        if match:
            files.append((int(match.group(1)), path))

    files.sort(key=lambda item: item[0])

    return [path for _, path in files]


def ensure_yxt(stack):
    """
    Ensure FLIM stack has shape Y, X, T.

    Raw TIFFs may be stored as:
        T, Y, X

    or already as:
        Y, X, T

    This function returns:
        Y, X, T
    """
    arr = np.asarray(stack)

    if arr.ndim != 3:
        raise ValueError(f"Esperaba stack 3D, recibí shape {arr.shape}")

    if arr.shape[0] == N_TOTAL:
        arr = np.moveaxis(arr, 0, -1)

    elif arr.shape[-1] == N_TOTAL:
        pass

    else:
        raise ValueError(
            f"Esperaba {N_TOTAL} canales en eje 0 o eje -1, "
            f"pero recibí shape {arr.shape}"
        )

    return arr.astype(np.float32, copy=False)


def flim_to_rgb_raw(stack):
    """
    Convert one FLIM decay stack into raw R/G/B channels.

    Input:
        stack with shape T,Y,X or Y,X,T

    Output:
        R, G, B as float32 2D images.
    """
    stack_yxt = ensure_yxt(stack)

    green = stack_yxt[..., :N_GREEN]
    blue = stack_yxt[..., N_GREEN:]

    red_channel = green[..., :2].sum(axis=-1).astype(np.float32)
    green_channel = green[..., 2:].sum(axis=-1).astype(np.float32)
    blue_channel = blue.sum(axis=-1).astype(np.float32)

    red_channel *= SCALE_R
    green_channel *= SCALE_G
    blue_channel *= SCALE_B

    return red_channel, green_channel, blue_channel


def normalize_channel(channel, low_value, high_value):
    """
    Normalize one channel using fixed low/high values.
    """
    if high_value <= low_value:
        return np.zeros_like(channel, dtype=np.float32)

    normalized = np.clip(channel, low_value, high_value)
    normalized = (normalized - low_value) / (high_value - low_value)

    return normalized.astype(np.float32)


def rgb_float_to_uint8(rgb):
    """
    Convert RGB float image in [0, 1] to uint8.
    """
    rgb = np.clip(rgb, 0, 1)
    return (rgb * 255).round().astype(np.uint8)


def save_png(rgb_uint8, output_path):
    """
    Save RGB uint8 image as PNG with DPI metadata.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.fromarray(rgb_uint8)
    image.save(output_path, dpi=(PNG_DPI, PNG_DPI), optimize=True)


def snake_indices(rows, cols):
    """
    Return tile index layout for snake acquisition.

    Example for 4x4:
        row 0:  0,  1,  2,  3
        row 1:  7,  6,  5,  4
        row 2:  8,  9, 10, 11
        row 3: 15, 14, 13, 12
    """
    indices = np.arange(rows * cols).reshape(rows, cols)
    layout = []

    for row in range(rows):
        row_indices = indices[row].copy()

        if row % 2 == 1:
            row_indices = row_indices[::-1]

        layout.append(row_indices.tolist())

    return layout


def reconstruct_snake_mosaic(tile_images, rows, cols):
    """
    Reconstruct RGB mosaic using snake acquisition layout.
    """
    expected = rows * cols

    if len(tile_images) != expected:
        raise ValueError(f"Esperaba {expected} tiles, recibí {len(tile_images)}")

    height, width, channels = tile_images[0].shape

    if channels != 3:
        raise ValueError(f"Esperaba imágenes RGB, recibí {channels} canales")

    mosaic = np.zeros(
        (rows * height, cols * width, channels),
        dtype=np.uint8,
    )

    layout = snake_indices(rows, cols)

    for row in range(rows):
        for col in range(cols):
            tile_index = layout[row][col]
            tile = tile_images[tile_index]

            y0 = row * height
            y1 = y0 + height
            x0 = col * width
            x1 = x0 + width

            mosaic[y0:y1, x0:x1, :] = tile

    return mosaic


def write_metadata(
    metadata_path,
    mosaic_dir,
    flim_dir,
    tile_paths,
    percentiles,
    rows,
    cols,
):
    """
    Write metadata describing RGB conversion.
    """
    r_low, r_high = percentiles["R"]
    g_low, g_high = percentiles["G"]
    b_low, b_high = percentiles["B"]

    text = f"""FLIM-to-RGB metadata

Patient folder:
{PATIENT_FOLDER}

Mosaic folder:
{mosaic_dir}

FLIM folder:
{flim_dir}

Mosaic shape:
{rows} rows x {cols} columns

Detector split:
green detector = first {N_GREEN} bins
blue detector = remaining bins

Expected total bins:
{N_TOTAL}

RGB mapping:
R = sum green bins 0:2
G = sum green bins 2:{N_GREEN}
B = sum blue bins {N_GREEN}:{N_TOTAL}

Channel scaling:
R scale = {SCALE_R}
G scale = {SCALE_G}
B scale = {SCALE_B}

Percentile normalization:
low percentile = {PERCENTILE_LOW}
high percentile = {PERCENTILE_HIGH}

Percentile values:
R = {r_low} to {r_high}
G = {g_low} to {g_high}
B = {b_low} to {b_high}

PNG DPI:
{PNG_DPI}

Tiles used:
"""

    for path in tile_paths:
        text += f"- {path.name}\n"

    metadata_path.write_text(text)


# =========================
# PROCESSING
# =========================


def process_mosaic_folder(mosaic_dir):
    """
    Process one mosaic folder.
    """
    flim_dir = mosaic_dir / FLIM_SUBDIR
    rgb_out_dir = mosaic_dir / RGB_OUTPUT_SUBDIR

    if not flim_dir.exists():
        print(f"  [SKIP] No existe flim/: {flim_dir}")
        return

    rows, cols = parse_mosaic_shape_from_name(mosaic_dir.name)
    expected_tiles = rows * cols

    tile_paths = collect_flim_tile_paths(flim_dir)

    if len(tile_paths) == 0:
        print(f"  [SKIP] No hay tiles Im_*.tif en: {flim_dir}")
        return

    if len(tile_paths) != expected_tiles:
        print(
            f"  [WARN] {mosaic_dir.name}: esperaba {expected_tiles} tiles, "
            f"pero encontré {len(tile_paths)}."
        )

    rgb_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcesando mosaico: {mosaic_dir.name}")
    print(f"  FLIM dir: {flim_dir}")
    print(f"  RGB dir:  {rgb_out_dir}")
    print(f"  Tiles encontrados: {len(tile_paths)}")
    print(f"  Mosaico detectado: {rows}x{cols}")

    raw_r_tiles = []
    raw_g_tiles = []
    raw_b_tiles = []
    valid_tile_paths = []
    rgb_raw_tiles = []

    for tile_path in tile_paths:
        try:
            stack = tiff.imread(tile_path)
            red, green, blue = flim_to_rgb_raw(stack)

            raw_r_tiles.append(red)
            raw_g_tiles.append(green)
            raw_b_tiles.append(blue)

            rgb_raw_tiles.append((red, green, blue))
            valid_tile_paths.append(tile_path)

        except Exception as exc:
            print(f"  [ERROR] Leyendo {tile_path.name}: {exc}")

    if len(valid_tile_paths) == 0:
        print("  [SKIP] No se pudo procesar ningún tile válido.")
        return

    all_r = np.concatenate([tile.ravel() for tile in raw_r_tiles])
    all_g = np.concatenate([tile.ravel() for tile in raw_g_tiles])
    all_b = np.concatenate([tile.ravel() for tile in raw_b_tiles])

    r_low, r_high = np.percentile(all_r, (PERCENTILE_LOW, PERCENTILE_HIGH))
    g_low, g_high = np.percentile(all_g, (PERCENTILE_LOW, PERCENTILE_HIGH))
    b_low, b_high = np.percentile(all_b, (PERCENTILE_LOW, PERCENTILE_HIGH))

    percentiles = {
        "R": (float(r_low), float(r_high)),
        "G": (float(g_low), float(g_high)),
        "B": (float(b_low), float(b_high)),
    }

    print(f"  Percentiles R: {r_low:.2f} - {r_high:.2f}")
    print(f"  Percentiles G: {g_low:.2f} - {g_high:.2f}")
    print(f"  Percentiles B: {b_low:.2f} - {b_high:.2f}")

    rgb_tiles = []

    for tile_path, (red, green, blue) in zip(valid_tile_paths, rgb_raw_tiles):
        try:
            out_png = rgb_out_dir / f"{tile_path.stem}_RGB.png"

            if out_png.exists() and not OVERWRITE:
                print(f"  [SKIP] Existe: {out_png.name}")
                continue

            red_norm = normalize_channel(red, r_low, r_high)
            green_norm = normalize_channel(green, g_low, g_high)
            blue_norm = normalize_channel(blue, b_low, b_high)

            rgb_float = np.stack(
                [red_norm, green_norm, blue_norm],
                axis=-1,
            )

            rgb_uint8 = rgb_float_to_uint8(rgb_float)

            save_png(rgb_uint8, out_png)
            rgb_tiles.append(rgb_uint8)

        except Exception as exc:
            print(f"  [ERROR] Guardando {tile_path.name}: {exc}")

    metadata_path = rgb_out_dir / "rgb_metadata.txt"
    write_metadata(
        metadata_path=metadata_path,
        mosaic_dir=mosaic_dir,
        flim_dir=flim_dir,
        tile_paths=valid_tile_paths,
        percentiles=percentiles,
        rows=rows,
        cols=cols,
    )

    print(f"  Metadata guardada: {metadata_path}")

    if len(rgb_tiles) == expected_tiles:
        try:
            mosaic = reconstruct_snake_mosaic(
                rgb_tiles,
                rows=rows,
                cols=cols,
            )

            mosaic_png = rgb_out_dir / f"{mosaic_dir.name}_RGB_mosaic.png"

            if mosaic_png.exists() and not OVERWRITE:
                print(f"  [SKIP] Existe mosaico: {mosaic_png.name}")
            else:
                save_png(mosaic, mosaic_png)
                print("  Mosaico RGB reconstruido guardado:")
                print(f"    {mosaic_png}")

        except Exception as exc:
            print(f"  [ERROR] Reconstruyendo mosaico: {exc}")

    else:
        print(
            f"  [WARN] No se reconstruyó mosaico: hay {len(rgb_tiles)} tiles "
            f"RGB válidos y se esperaban {expected_tiles}."
        )


def main():
    """
    Process all visits and mosaics inside PATIENT_FOLDER.
    """
    patient_dir = PATIENT_FOLDER

    if not patient_dir.exists():
        raise FileNotFoundError(f"No existe PATIENT_FOLDER:\n{patient_dir}")

    visit_dirs = sorted(
        [
            path
            for path in patient_dir.iterdir()
            if path.is_dir() and path.name.lower().startswith("visit")
        ],
        key=natural_key,
    )

    print(f"Paciente: {patient_dir}")
    print(f"Visitas encontradas: {[visit.name for visit in visit_dirs]}")

    if len(visit_dirs) == 0:
        print("[WARN] No se encontraron carpetas visit*.")
        return

    for visit_dir in visit_dirs:
        print("\n==============================")
        print(f"Procesando visita: {visit_dir.name}")
        print("==============================")

        mosaic_dirs = sorted(
            [
                path
                for path in visit_dir.iterdir()
                if path.is_dir() and path.name.lower().startswith("mosaic")
            ],
            key=natural_key,
        )

        print(f"Mosaicos encontrados: {len(mosaic_dirs)}")

        for mosaic_dir in mosaic_dirs:
            process_mosaic_folder(mosaic_dir)

    print("\nListo.")


if __name__ == "__main__":
    main()
