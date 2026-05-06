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


# =========================
# CONFIG
# =========================

PATIENT_DIR = Path(
    "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449"
).expanduser()

RGB_SUBDIR = Path("RGB")
MANUAL_MASK_SUBDIR = Path("random_forest/mask")

OUTPUT_DIR = PATIENT_DIR / "QC_manual_mask_rgb_overlay_PDFs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RGB_EXTENSIONS = [".png"]
MASK_EXTENSIONS = [".png", ".tif", ".tiff"]

PDF_DPI = 600
OVERLAY_ALPHA = 0.20
BOUNDARY_COLOR = np.array([1.0, 1.0, 1.0])  # white


# =========================
# HELPERS
# =========================

def natural_key(path):
    return [
        int(t) if t.isdigit() else t.lower()
        for t in re.split(r"(\d+)", str(path.name))
    ]


def extract_tile_number(name):
    patterns = [
        r"Im_(\d+)",
        r"_t(\d+)",
        r"tile[_-]?(\d+)",
    ]

    for pat in patterns:
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


def find_manual_mask_files(mosaic_dir):
    mask_dir = mosaic_dir / MANUAL_MASK_SUBDIR

    if not mask_dir.exists():
        return []

    files = []
    for ext in MASK_EXTENSIONS:
        files.extend(mask_dir.glob(f"*{ext}"))

    return sorted(files, key=natural_key)


def match_tile_file(tile_number, candidate_files):
    if tile_number is None:
        return None

    for f in candidate_files:
        if extract_tile_number(f.name) == tile_number:
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

    for ax in axes:
        ax.axis("off")

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
    visit_dirs = sorted(
        [p for p in PATIENT_DIR.glob("visit*") if p.is_dir()],
        key=natural_key,
    )

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
                [p for p in visit_dir.glob("Mosaic*") if p.is_dir()],
                key=natural_key,
            )

            for mosaic_dir in mosaic_dirs:
                mosaic_name = mosaic_dir.name

                rgb_files = find_rgb_files(mosaic_dir)
                manual_mask_files = find_manual_mask_files(mosaic_dir)

                if len(manual_mask_files) == 0:
                    continue

                if len(rgb_files) == 0:
                    print(f"[WARNING] No RGB files found in {mosaic_dir / RGB_SUBDIR}")
                    missing_rgb.extend(manual_mask_files)
                    continue

                for mask_file in manual_mask_files:
                    tile_number = extract_tile_number(mask_file.name)
                    rgb_file = match_tile_file(tile_number, rgb_files)

                    if rgb_file is None:
                        print(f"[WARNING] No RGB match for manual mask: {mask_file}")
                        missing_rgb.append(mask_file)
                        continue

                    rgb = read_rgb(rgb_file)
                    mask = read_mask(mask_file)
                    overlay, mask = make_overlay(rgb, mask)

                    tile_label = (
                        f"{tile_number:02d}"
                        if tile_number is not None
                        else "unknown"
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

    if missing_rgb:
        print(f"\nWARNING: {len(missing_rgb)} manual masks had no matching RGB.")
        for f in missing_rgb[:30]:
            print(f"Missing RGB for: {f}")
        if len(missing_rgb) > 30:
            print("...")


if __name__ == "__main__":
    main()