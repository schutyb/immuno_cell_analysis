#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import tifffile as tiff
from skimage.transform import resize
from skimage.segmentation import find_boundaries


PATIENT_DIR = Path(
    "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449"
).expanduser()

OUTPUT_DIR = PATIENT_DIR / "QC_original_vs_area_flim_masks_PDFs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RGB_SUBDIR = Path("RGB")
SEG_SUBDIR = Path("SegData")
FINAL_MASK_SUBDIR = Path("segmentation_area_phasor/tiles")

RGB_EXTENSIONS = [".png"]
MASK_EXTENSIONS = [".png", ".tif", ".tiff"]

PDF_DPI = 600
OVERLAY_ALPHA = 0.20
BOUNDARY_COLOR = np.array([1.0, 1.0, 1.0])


def natural_key(path):
    return [
        int(t) if t.isdigit() else t.lower()
        for t in re.split(r"(\d+)", str(path.name))
    ]


def extract_tile_number(name):
    for pat in [r"Im_(\d+)", r"_t(\d+)", r"tile[_-]?(\d+)"]:
        m = re.search(pat, name, re.IGNORECASE)
        if m:
            return int(m.group(1))

    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else None


def read_rgb(path):
    img = np.array(Image.open(path))

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    if img.shape[-1] == 4:
        img = img[..., :3]

    img = img.astype(np.float32)

    if img.max() > 1:
        img /= 255.0 if img.max() <= 255 else img.max()

    return np.clip(img, 0, 1)


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
    for ext in RGB_EXTENSIONS:
        files.extend(rgb_dir.glob(f"*{ext}"))

    files = [f for f in files if "mosaic" not in f.name.lower()]
    files = [f for f in files if "rgb" in f.name.lower()]

    return sorted(files, key=lambda f: extract_tile_number(f.name) or 9999)


def find_original_mask_files(mosaic_dir):
    seg_dir = mosaic_dir / SEG_SUBDIR
    if not seg_dir.exists():
        return []

    files = []
    for ext in MASK_EXTENSIONS:
        files.extend(seg_dir.glob(f"*{ext}"))

    return sorted(files, key=natural_key)


def find_final_mask_files(mosaic_dir):
    final_dir = mosaic_dir / FINAL_MASK_SUBDIR
    if not final_dir.exists():
        return []

    files = []
    for ext in [".tif", ".tiff"]:
        files.extend(final_dir.glob(f"*cell_mask_final*{ext}"))

    return sorted(files, key=natural_key)


def match_tile_file(rgb_file, candidate_files):
    rgb_tile = extract_tile_number(rgb_file.name)
    if rgb_tile is None:
        return None

    for f in candidate_files:
        if extract_tile_number(f.name) == rgb_tile:
            return f

    return None


def make_overlay(rgb, mask):
    if mask.shape[:2] != rgb.shape[:2]:
        mask = resize(
            mask.astype(float),
            rgb.shape[:2],
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ) > 0.5

    overlay = rgb.copy()

    overlay[mask] = (
        (1 - OVERLAY_ALPHA) * overlay[mask]
        + OVERLAY_ALPHA * BOUNDARY_COLOR
    )

    boundaries = find_boundaries(mask, mode="outer")
    overlay[boundaries] = BOUNDARY_COLOR

    return overlay, mask


def plot_comparison_page(
    pdf,
    original_mask,
    original_overlay,
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

    axes[0, 0].imshow(original_mask, cmap="gray", interpolation="none", vmin=0, vmax=1)
    axes[0, 0].set_title("Peter Chang mask", fontsize=9)

    axes[0, 1].imshow(original_overlay, interpolation="none")
    axes[0, 1].set_title("RGB + Peter Chang overlay", fontsize=9)

    axes[1, 0].imshow(final_mask, cmap="gray", interpolation="none", vmin=0, vmax=1)
    axes[1, 0].set_title("Area + FLIM filtered mask", fontsize=9)

    axes[1, 1].imshow(final_overlay, interpolation="none")
    axes[1, 1].set_title("RGB + area/FLIM overlay", fontsize=9)

    for ax in axes.ravel():
        ax.axis("off")

    pdf.savefig(fig, dpi=PDF_DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main():
    visit_dirs = sorted(
        [p for p in PATIENT_DIR.glob("visit*") if p.is_dir()],
        key=natural_key,
    )

    missing_original_masks = []
    missing_final_masks = []

    for visit_dir in visit_dirs:
        visit_name = visit_dir.name

        output_pdf = (
            OUTPUT_DIR
            / f"{PATIENT_DIR.name}_{visit_name}_original_vs_area_flim_final_masks_QC_600dpi.pdf"
        )

        page_count = 0

        with PdfPages(output_pdf) as pdf:
            mosaic_dirs = sorted(
                [p for p in visit_dir.glob("Mosaic*") if p.is_dir()],
                key=natural_key,
            )

            for mosaic_dir in mosaic_dirs:
                mosaic_name = mosaic_dir.name

                rgb_files = find_rgb_files(mosaic_dir)
                original_mask_files = find_original_mask_files(mosaic_dir)
                final_mask_files = find_final_mask_files(mosaic_dir)

                if len(rgb_files) == 0:
                    print(f"[WARNING] No RGB files found in {mosaic_dir / RGB_SUBDIR}")
                    continue

                if len(original_mask_files) == 0:
                    print(f"[WARNING] No original masks found in {mosaic_dir / SEG_SUBDIR}")

                if len(final_mask_files) == 0:
                    print(f"[WARNING] No final masks found in {mosaic_dir / FINAL_MASK_SUBDIR}")

                for rgb_file in rgb_files:
                    tile_number = extract_tile_number(rgb_file.name)

                    original_mask_file = match_tile_file(rgb_file, original_mask_files)
                    final_mask_file = match_tile_file(rgb_file, final_mask_files)

                    rgb = read_rgb(rgb_file)

                    if original_mask_file is not None:
                        original_mask = read_mask(original_mask_file)
                    else:
                        original_mask = np.zeros(rgb.shape[:2], dtype=bool)
                        missing_original_masks.append(rgb_file)

                    if final_mask_file is not None:
                        final_mask = read_mask(final_mask_file)
                    else:
                        final_mask = np.zeros(rgb.shape[:2], dtype=bool)
                        missing_final_masks.append(rgb_file)

                    original_overlay, original_mask = make_overlay(rgb, original_mask)
                    final_overlay, final_mask = make_overlay(rgb, final_mask)

                    tile_label = f"{tile_number:02d}" if tile_number is not None else "unknown"

                    title = (
                        f"{visit_name} | {mosaic_name} | Tile {tile_label}\n"
                        f"RGB: {rgb_file.name}\n"
                        f"Peter Chang mask: "
                        f"{original_mask_file.name if original_mask_file else 'NOT FOUND'} | "
                        f"Area + FLIM mask: "
                        f"{final_mask_file.name if final_mask_file else 'NOT FOUND'}"
                    )

                    plot_comparison_page(
                        pdf=pdf,
                        original_mask=original_mask,
                        original_overlay=original_overlay,
                        final_mask=final_mask,
                        final_overlay=final_overlay,
                        title=title,
                    )

                    page_count += 1

        print(f"[DONE] {visit_name}: {page_count} pages")
        print(f"Saved PDF:\n{output_pdf}\n")

    if missing_original_masks:
        print(f"\nWARNING: {len(missing_original_masks)} RGB tiles had no original mask.")

    if missing_final_masks:
        print(f"\nWARNING: {len(missing_final_masks)} RGB tiles had no final mask.")


if __name__ == "__main__":
    main()