#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create color overlays comparing manual masks against final cell masks.

This script compares manual expert cell masks with the final segmentation masks
produced by the area + phasor/GMM filtering pipeline.

The output images contain only the masks, not the RGB background:

    manual only = green
    final only  = red
    overlap     = yellow

This is useful for visual QC of agreement between manual annotations and the
final cell masks used for downstream immuno-cell analysis.

Input expected per mosaic:
    <mosaic_dir>/<MANUAL_MASK_SUBDIR>/
        manual mask files matching tile numbers

    <mosaic_dir>/<FINAL_MASK_SUBDIR>/
        Im_00001_cell_mask_final.tif
        Im_00002_cell_mask_final.tif
        ...

Output:
    <patient_dir>/segmentation_evaluation/manual_vs_final_mask_overlay_masks/
        visitXX/
            MosaicXX.../
                Im_00001_manual_green_final_red_overlap.png
                ...

Also writes:
    manual_vs_final_mask_overlay_index.csv

How to use:
    1. Edit PATIENT_DIR.
    2. Edit MANUAL_MASK_SUBDIR to point to the manual masks.
    3. Edit FINAL_MASK_SUBDIR if the final mask output folder changes.
    4. Run:

        python -m src.segmentation.export_manual_vs_final_mask_overlay

    or directly:

        python src/segmentation/export_manual_vs_final_mask_overlay.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
from PIL import Image
from skimage.transform import resize

# =========================
# CONFIG
# =========================

PATIENT_DIR = Path(
    "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449"
).expanduser()

# Temporary/current location.
# Recommended future location: Path("manual_masks")
MANUAL_MASK_SUBDIR = Path("random_forest/mask")

FINAL_MASK_SUBDIR = Path("final_masks_area_phasor_gmm/tiles")

OUTPUT_SUBDIR = Path("segmentation_evaluation/manual_vs_final_mask_overlay_masks")

MASK_EXTENSIONS = [".png", ".tif", ".tiff"]

MANUAL_COLOR = np.array([0.25, 0.85, 0.35], dtype=np.float32)
FINAL_COLOR = np.array([1.00, 0.25, 0.25], dtype=np.float32)
OVERLAP_COLOR = np.array([1.00, 0.90, 0.20], dtype=np.float32)
BACKGROUND_COLOR = np.array([0.0, 0.0, 0.0], dtype=np.float32)

SAVE_DPI = (600, 600)


# =========================
# HELPERS
# =========================


def natural_key(path):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", str(path.name))
    ]


def extract_tile_number(name):
    patterns = [
        r"Im_(\d+)",
        r"_t(\d+)",
        r"tile[_-]?(\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, name, re.IGNORECASE)

        if match:
            return int(match.group(1))

    numbers = re.findall(r"\d+", name)

    return int(numbers[-1]) if numbers else None


def read_mask(path):
    if path.suffix.lower() in [".tif", ".tiff"]:
        mask = tiff.imread(path)
    else:
        mask = np.array(Image.open(path))

    if mask.ndim == 3:
        mask = mask[..., 0]

    return mask > 0


def resize_mask_if_needed(mask, target_shape):
    if mask.shape == target_shape:
        return mask

    resized = resize(
        mask.astype(float),
        target_shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )

    return resized > 0.5


def collect_mask_files(folder):
    if not folder.exists():
        return []

    files = []

    for extension in MASK_EXTENSIONS:
        files.extend(folder.glob(f"*{extension}"))

    return sorted(files, key=natural_key)


def match_by_tile(tile_number, files):
    for path in files:
        if extract_tile_number(path.name) == tile_number:
            return path

    return None


def make_overlay_mask(manual_mask, final_mask):
    final_mask = resize_mask_if_needed(final_mask, manual_mask.shape)

    rgb = np.zeros((*manual_mask.shape, 3), dtype=np.float32)
    rgb[...] = BACKGROUND_COLOR

    manual_only = manual_mask & ~final_mask
    final_only = final_mask & ~manual_mask
    overlap = manual_mask & final_mask

    rgb[manual_only] = MANUAL_COLOR
    rgb[final_only] = FINAL_COLOR
    rgb[overlap] = OVERLAP_COLOR

    return np.clip(rgb, 0, 1)


def save_png(rgb_float, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rgb_uint8 = (np.clip(rgb_float, 0, 1) * 255).round().astype(np.uint8)
    Image.fromarray(rgb_uint8).save(output_path, dpi=SAVE_DPI)


# =========================
# MAIN
# =========================


def main():
    if not PATIENT_DIR.exists():
        raise FileNotFoundError(f"No existe PATIENT_DIR:\n{PATIENT_DIR}")

    output_dir = PATIENT_DIR / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    visit_dirs = sorted(
        [path for path in PATIENT_DIR.glob("visit*") if path.is_dir()],
        key=natural_key,
    )

    total_saved = 0
    missing_final = 0

    for visit_dir in visit_dirs:
        visit_name = visit_dir.name

        mosaic_dirs = sorted(
            [path for path in visit_dir.glob("Mosaic*") if path.is_dir()],
            key=natural_key,
        )

        for mosaic_dir in mosaic_dirs:
            mosaic_name = mosaic_dir.name

            manual_dir = mosaic_dir / MANUAL_MASK_SUBDIR
            final_dir = mosaic_dir / FINAL_MASK_SUBDIR

            manual_files = collect_mask_files(manual_dir)
            final_files = collect_mask_files(final_dir)

            if len(manual_files) == 0:
                continue

            mosaic_out_dir = output_dir / visit_name / mosaic_name
            mosaic_out_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n[PROCESS] {visit_name} | {mosaic_name}")
            print(f"Manual masks: {len(manual_files)}")
            print(f"Final masks:  {len(final_files)}")

            for manual_file in manual_files:
                tile_number = extract_tile_number(manual_file.name)

                if tile_number is None:
                    print(f"[WARN] Could not extract tile number: {manual_file.name}")
                    continue

                final_file = match_by_tile(tile_number, final_files)

                if final_file is None:
                    print(f"[WARN] Missing final mask for tile {tile_number}")
                    missing_final += 1
                    continue

                manual_mask = read_mask(manual_file)
                final_mask = read_mask(final_file)
                final_mask = resize_mask_if_needed(final_mask, manual_mask.shape)

                overlay = make_overlay_mask(manual_mask, final_mask)

                tile_label = f"{tile_number:05d}"
                output_name = f"Im_{tile_label}_manual_green_final_red_overlap.png"
                output_path = mosaic_out_dir / output_name

                save_png(overlay, output_path)

                rows.append(
                    {
                        "visit": visit_name,
                        "mosaic": mosaic_name,
                        "tile": tile_number,
                        "manual_mask": str(manual_file),
                        "final_mask": str(final_file),
                        "overlay_mask": str(output_path),
                        "manual_px": int(manual_mask.sum()),
                        "final_px": int(final_mask.sum()),
                        "overlap_px": int((manual_mask & final_mask).sum()),
                    }
                )

                total_saved += 1

    if len(rows) > 0:
        csv_path = output_dir / "manual_vs_final_mask_overlay_index.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"\nIndex CSV:\n{csv_path}")

    print("\n[DONE]")
    print(f"Saved overlays: {total_saved}")
    print(f"Missing final masks: {missing_final}")
    print(f"Output folder:\n{output_dir}")


if __name__ == "__main__":
    main()
