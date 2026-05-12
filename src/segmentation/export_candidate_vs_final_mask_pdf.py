#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create QC PDFs comparing candidate masks against final filtered cell masks.

This script compares the original candidate segmentation masks from SegData/
with the final cell masks produced after area + phasor/GMM filtering.

For each RGB tile, the PDF shows:

    1. candidate SegData mask
    2. RGB + candidate mask overlay
    3. final area/phasor/GMM filtered cell mask
    4. RGB + final mask overlay

This is useful to visually verify which candidate objects were removed by the
final filtering step.

Input expected per mosaic:
    <mosaic_dir>/RGB/
        Im_00001_RGB.png
        ...

    <mosaic_dir>/SegData/
        candidate segmentation masks

    <mosaic_dir>/final_masks_area_phasor_gmm/tiles/
        Im_00001_cell_mask_final.tif
        ...

Output:
    <patient_dir>/QC_candidate_vs_final_mask_PDFs/
        <patient>_<visit>_candidate_vs_final_masks_QC_600dpi.pdf

How to use:
    1. Edit PATIENT_DIR.
    2. Make sure RGB images exist.
    3. Make sure SegData candidate masks exist.
    4. Run build_area_phasor_cell_mask.py first to generate final masks.
    5. Run:

        python -m src.segmentation.export_candidate_vs_final_mask_pdf

    or directly:

        python src/segmentation/export_candidate_vs_final_mask_pdf.py
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

OUTPUT_DIR = PATIENT_DIR / "QC_candidate_vs_final_mask_PDFs"

RGB_SUBDIR = Path("RGB")
SEG_SUBDIR = Path("SegData")
FINAL_MASK_SUBDIR = Path("final_masks_area_phasor_gmm/tiles")

RGB_EXTENSIONS = [".png"]
MASK_EXTENSIONS = [".png", ".tif", ".tiff"]

PDF_DPI = 600
OVERLAY_ALPHA = 0.20
OVERLAY_COLOR = np.array([1.0, 1.0, 1.0])
BOUNDARY_COLOR = np.array([1.0, 1.0, 1.0])


# =========================
# HELPERS
# =========================


def natural_key(path):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", str(path.name))
    ]


def extract_tile_number(name):
    for pattern in [r"Im_(\d+)", r"_t(\d+)", r"tile[_-]?(\d+)"]:
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


def find_candidate_mask_files(mosaic_dir):
    seg_dir = mosaic_dir / SEG_SUBDIR

    if not seg_dir.exists():
        return []

    files = []

    for extension in MASK_EXTENSIONS:
        files.extend(seg_dir.glob(f"*{extension}"))

    return sorted(files, key=natural_key)


def find_final_mask_files(mosaic_dir):
    final_dir = mosaic_dir / FINAL_MASK_SUBDIR

    if not final_dir.exists():
        return []

    files = []

    for extension in [".tif", ".tiff"]:
        files.extend(final_dir.glob(f"*cell_mask_final*{extension}"))

    return sorted(files, key=natural_key)


def match_tile_file(rgb_file, candidate_files):
    rgb_tile_number = extract_tile_number(rgb_file.name)

    if rgb_tile_number is None:
        return None

    for path in candidate_files:
        if extract_tile_number(path.name) == rgb_tile_number:
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


def plot_comparison_page(
    pdf,
    candidate_mask,
    candidate_overlay,
    final_mask,
    final_overlay,
    title,
):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(9.5, 9.0),
        dpi=PDF_DPI,
        constrained_layout=True,
    )

    fig.suptitle(title, fontsize=10)

    axes[0, 0].imshow(
        candidate_mask,
        cmap="gray",
        interpolation="none",
        vmin=0,
        vmax=1,
    )
    axes[0, 0].set_title("Candidate SegData mask", fontsize=9)

    axes[0, 1].imshow(candidate_overlay, interpolation="none")
    axes[0, 1].set_title("RGB + candidate overlay", fontsize=9)

    axes[1, 0].imshow(
        final_mask,
        cmap="gray",
        interpolation="none",
        vmin=0,
        vmax=1,
    )
    axes[1, 0].set_title("Final area + phasor/GMM mask", fontsize=9)

    axes[1, 1].imshow(final_overlay, interpolation="none")
    axes[1, 1].set_title("RGB + final overlay", fontsize=9)

    for axis in axes.ravel():
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

    missing_candidate_masks = []
    missing_final_masks = []

    for visit_dir in visit_dirs:
        visit_name = visit_dir.name

        output_pdf = (
            OUTPUT_DIR
            / f"{PATIENT_DIR.name}_{visit_name}_candidate_vs_final_masks_QC_600dpi.pdf"
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
                candidate_mask_files = find_candidate_mask_files(mosaic_dir)
                final_mask_files = find_final_mask_files(mosaic_dir)

                if len(rgb_files) == 0:
                    print(f"[WARN] No RGB files found in {mosaic_dir / RGB_SUBDIR}")
                    continue

                if len(candidate_mask_files) == 0:
                    print(
                        f"[WARN] No candidate masks found in {mosaic_dir / SEG_SUBDIR}"
                    )

                if len(final_mask_files) == 0:
                    print(
                        f"[WARN] No final masks found in "
                        f"{mosaic_dir / FINAL_MASK_SUBDIR}"
                    )

                for rgb_file in rgb_files:
                    tile_number = extract_tile_number(rgb_file.name)

                    candidate_mask_file = match_tile_file(
                        rgb_file,
                        candidate_mask_files,
                    )
                    final_mask_file = match_tile_file(
                        rgb_file,
                        final_mask_files,
                    )

                    rgb = read_rgb(rgb_file)

                    if candidate_mask_file is not None:
                        candidate_mask = read_mask(candidate_mask_file)
                    else:
                        candidate_mask = np.zeros(rgb.shape[:2], dtype=bool)
                        missing_candidate_masks.append(rgb_file)

                    if final_mask_file is not None:
                        final_mask = read_mask(final_mask_file)
                    else:
                        final_mask = np.zeros(rgb.shape[:2], dtype=bool)
                        missing_final_masks.append(rgb_file)

                    candidate_overlay, candidate_mask = make_overlay(
                        rgb,
                        candidate_mask,
                    )
                    final_overlay, final_mask = make_overlay(
                        rgb,
                        final_mask,
                    )

                    tile_label = (
                        f"{tile_number:05d}" if tile_number is not None else "unknown"
                    )

                    title = (
                        f"{visit_name} | {mosaic_name} | Tile {tile_label}\n"
                        f"RGB: {rgb_file.name}\n"
                        f"Candidate mask: "
                        f"{candidate_mask_file.name if candidate_mask_file else 'NOT FOUND'} | "
                        f"Final mask: "
                        f"{final_mask_file.name if final_mask_file else 'NOT FOUND'}"
                    )

                    plot_comparison_page(
                        pdf=pdf,
                        candidate_mask=candidate_mask,
                        candidate_overlay=candidate_overlay,
                        final_mask=final_mask,
                        final_overlay=final_overlay,
                        title=title,
                    )

                    page_count += 1

        if page_count > 0:
            print(f"[DONE] {visit_name}: {page_count} pages")
            print(f"Saved PDF:\n{output_pdf}\n")
        else:
            output_pdf.unlink(missing_ok=True)

    if len(missing_candidate_masks) > 0:
        print(
            f"\n[WARN] {len(missing_candidate_masks)} RGB tiles had no candidate mask."
        )

    if len(missing_final_masks) > 0:
        print(f"\n[WARN] {len(missing_final_masks)} RGB tiles had no final mask.")


if __name__ == "__main__":
    main()
