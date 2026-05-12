#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create QC PDFs showing RGB tiles, manual cell masks, and overlays.

This script is used to visually inspect manual expert annotations against the
FLIM-derived RGB tiles.

For each patient visit, the script creates one PDF containing all available
manual masks and their matching RGB tiles.

Input expected per mosaic:
    <mosaic_dir>/RGB/
        Im_00001_RGB.png
        Im_00002_RGB.png
        ...

    <mosaic_dir>/<MANUAL_MASK_SUBDIR>/
        manual mask files matching tile numbers

Output:
    <patient_dir>/QC_manual_mask_rgb_overlay_PDFs/
        <patient>_<visit>_manual_mask_rgb_overlay_QC_600dpi.pdf

How to use:
    1. Edit PATIENT_DIR.
    2. Edit MANUAL_MASK_SUBDIR to point to the manual mask folder.
    3. Make sure RGB images were generated with src/utils/flim2rgb.py.
    4. Run from the repository root:

        python -m src.segmentation.export_rgb_manual_mask_overlay_pdf

    or directly:

        python src/segmentation/export_rgb_manual_mask_overlay_pdf.py
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from skimage.segmentation import find_boundaries
from skimage.transform import resize

# =========================
# CONFIG
# =========================

PATIENT_DIR = Path(
    "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449"
).expanduser()

RGB_SUBDIR = Path("RGB")

# Temporary/current location.
# Recommended future location: Path("manual_masks")
MANUAL_MASK_SUBDIR = Path("random_forest/mask")

OUTPUT_DIR = PATIENT_DIR / "QC_manual_mask_rgb_overlay_PDFs"

RGB_EXTENSIONS = [".png"]
MASK_EXTENSIONS = [".png", ".tif", ".tiff"]

PDF_DPI = 600

OVERLAY_ALPHA = 0.20
OVERLAY_COLOR = np.array([1.0, 1.0, 1.0])  # white
BOUNDARY_COLOR = np.array([1.0, 1.0, 1.0])  # white


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


def read_rgb(path):
    image = np.array(Image.open(path))

    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)

    if image.shape[-1] == 4:
        image = image[..., :3]

    image = image.astype(np.float32)

    if image.max() > 1:
        if image.max() <= 255:
            image /= 255.0
        else:
            image /= image.max()

    return np.clip(image, 0, 1)


def read_mask(path):
    if path.suffix.lower() in [".tif", ".tiff"]:
        mask = tiff.imread(path)
    else:
        mask = np.array(Image.open(path))

    if mask.ndim == 3:
        mask = mask[..., 0]

    return mask > 0


def find_rgb_files(mosaic_dir):
    rgb_dir = mosaic_dir / RGB_SUBDIR

    if not rgb_dir.exists():
        return []

    files = []

    for extension in RGB_EXTENSIONS:
        files.extend(rgb_dir.glob(f"*{extension}"))

    files = [path for path in files if "mosaic" not in path.name.lower()]
    files = [path for path in files if "rgb" in path.name.lower()]

    return sorted(
        files,
        key=lambda path: extract_tile_number(path.name) or 999999,
    )


def find_manual_mask_files(mosaic_dir):
    mask_dir = mosaic_dir / MANUAL_MASK_SUBDIR

    if not mask_dir.exists():
        return []

    files = []

    for extension in MASK_EXTENSIONS:
        files.extend(mask_dir.glob(f"*{extension}"))

    return sorted(files, key=natural_key)


def match_tile_file(tile_number, candidate_files):
    if tile_number is None:
        return None

    for path in candidate_files:
        if extract_tile_number(path.name) == tile_number:
            return path

    return None


def resize_mask_to_rgb(mask, rgb_shape):
    if mask.shape[:2] == rgb_shape[:2]:
        return mask

    resized = resize(
        mask.astype(float),
        rgb_shape[:2],
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )

    return resized > 0.5


def make_overlay(rgb, mask):
    mask = resize_mask_to_rgb(mask, rgb.shape)

    overlay = rgb.copy()

    overlay[mask] = (1 - OVERLAY_ALPHA) * overlay[mask] + OVERLAY_ALPHA * OVERLAY_COLOR

    boundaries = find_boundaries(mask, mode="outer")
    overlay[boundaries] = BOUNDARY_COLOR

    return overlay, mask


def plot_page(pdf, rgb, mask, overlay, title):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.5, 4.6),
        dpi=PDF_DPI,
        constrained_layout=True,
    )

    fig.suptitle(title, fontsize=10)

    axes[0].imshow(rgb, interpolation="none")
    axes[0].set_title("RGB tile", fontsize=9)

    axes[1].imshow(mask, cmap="gray", interpolation="none", vmin=0, vmax=1)
    axes[1].set_title("Manual cell mask", fontsize=9)

    axes[2].imshow(overlay, interpolation="none")
    axes[2].set_title("RGB + manual mask overlay", fontsize=9)

    for axis in axes:
        axis.axis("off")

    pdf.savefig(
        fig,
        dpi=PDF_DPI,
        bbox_inches="tight",
        pad_inches=0.08,
    )

    plt.close(fig)


# =========================
# MAIN
# =========================


def main():
    if not PATIENT_DIR.exists():
        raise FileNotFoundError(f"No existe PATIENT_DIR:\n{PATIENT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    visit_dirs = sorted(
        [path for path in PATIENT_DIR.glob("visit*") if path.is_dir()],
        key=natural_key,
    )

    if len(visit_dirs) == 0:
        print("[WARN] No se encontraron carpetas visit*.")
        return

    missing_rgb = []
    total_pages = 0

    for visit_dir in visit_dirs:
        visit_name = visit_dir.name

        output_pdf = (
            OUTPUT_DIR
            / f"{PATIENT_DIR.name}_{visit_name}_manual_mask_rgb_overlay_QC_600dpi.pdf"
        )

        page_count = 0

        with PdfPages(output_pdf) as pdf:
            mosaic_dirs = sorted(
                [path for path in visit_dir.glob("Mosaic*") if path.is_dir()],
                key=natural_key,
            )

            for mosaic_dir in mosaic_dirs:
                mosaic_name = mosaic_dir.name

                rgb_files = find_rgb_files(mosaic_dir)
                manual_mask_files = find_manual_mask_files(mosaic_dir)

                if len(manual_mask_files) == 0:
                    continue

                if len(rgb_files) == 0:
                    print(f"[WARN] No RGB files found in {mosaic_dir / RGB_SUBDIR}")
                    missing_rgb.extend(manual_mask_files)
                    continue

                for mask_file in manual_mask_files:
                    tile_number = extract_tile_number(mask_file.name)
                    rgb_file = match_tile_file(tile_number, rgb_files)

                    if rgb_file is None:
                        print(f"[WARN] No RGB match for manual mask: {mask_file}")
                        missing_rgb.append(mask_file)
                        continue

                    rgb = read_rgb(rgb_file)
                    mask = read_mask(mask_file)
                    overlay, mask = make_overlay(rgb, mask)

                    tile_label = (
                        f"{tile_number:05d}" if tile_number is not None else "unknown"
                    )

                    title = (
                        f"{visit_name} | {mosaic_name} | Tile {tile_label}\n"
                        f"RGB: {rgb_file.name}\n"
                        f"Manual mask: {mask_file.name}"
                    )

                    plot_page(
                        pdf=pdf,
                        rgb=rgb,
                        mask=mask,
                        overlay=overlay,
                        title=title,
                    )

                    page_count += 1
                    total_pages += 1

        if page_count > 0:
            print(f"[DONE] {visit_name}: {page_count} pages")
            print(f"Saved PDF:\n{output_pdf}\n")
        else:
            output_pdf.unlink(missing_ok=True)

    print("\nDone.")
    print(f"Total pages: {total_pages}")

    if len(missing_rgb) > 0:
        print(f"\n[WARN] {len(missing_rgb)} manual masks had no matching RGB.")

        for path in missing_rgb[:30]:
            print(f"Missing RGB for: {path}")

        if len(missing_rgb) > 30:
            print("...")


if __name__ == "__main__":
    main()
