#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from skimage.transform import resize
from skimage.segmentation import find_boundaries


# =========================
# CONFIG
# =========================

PATIENT_DIR = Path(
    "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449"
).expanduser()

OUTPUT_DIR = PATIENT_DIR / "QC_rgb_mask_overlay_PDFs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RGB_SUBDIR = Path("RGB")   # <-- actualizado
SEG_SUBDIR = Path("SegData")

RGB_EXTENSIONS = [".png"]  # <-- solo PNG
MASK_EXTENSIONS = [".png", ".tif", ".tiff"]

PDF_DPI = 600
OVERLAY_ALPHA = 0.2
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
        if img.max() <= 255:
            img /= 255.0
        else:
            img /= img.max()

    return np.clip(img, 0, 1)


def read_mask(path):
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

    # 🔥 ignorar mosaico reconstruido
    files = [f for f in files if "mosaic" not in f.name.lower()]

    # 🔥 opcional: asegurar que sean RGB generados
    files = [f for f in files if "rgb" in f.name.lower()]

    files = sorted(
        files,
        key=lambda f: extract_tile_number(f.name) or 9999
    )

    return files


def find_mask_files(mosaic_dir):
    seg_dir = mosaic_dir / SEG_SUBDIR

    if not seg_dir.exists():
        return []

    files = []
    for ext in MASK_EXTENSIONS:
        files.extend(seg_dir.glob(f"*{ext}"))

    return sorted(files, key=natural_key)


def match_mask(rgb_file, mask_files):
    rgb_tile = extract_tile_number(rgb_file.name)

    if rgb_tile is None:
        return None

    for mf in mask_files:
        mask_tile = extract_tile_number(mf.name)
        if mask_tile == rgb_tile:
            return mf

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


def plot_tile_page(pdf, rgb, mask, overlay, title):
    fig, axes = plt.subplots(
        1, 3,
        figsize=(13.5, 4.6),
        dpi=PDF_DPI,
        constrained_layout=True,
    )

    fig.suptitle(title, fontsize=10)

    axes[0].imshow(rgb, interpolation="none")
    axes[0].set_title("RGB tile", fontsize=9)

    axes[1].imshow(mask, cmap="gray", interpolation="none", vmin=0, vmax=1)
    axes[1].set_title("Peter Chang mask", fontsize=9)

    axes[2].imshow(overlay, interpolation="none")
    axes[2].set_title("Overlay", fontsize=9)

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

    missing_masks = []

    for visit_dir in visit_dirs:
        visit_name = visit_dir.name
        output_pdf = OUTPUT_DIR / f"{PATIENT_DIR.name}_{visit_name}_rgb_mask_overlay_QC_600dpi.pdf"

        page_count = 0

        with PdfPages(output_pdf) as pdf:
            mosaic_dirs = sorted(
                [p for p in visit_dir.glob("Mosaic*") if p.is_dir()],
                key=natural_key,
            )

            for mosaic_dir in mosaic_dirs:
                mosaic_name = mosaic_dir.name

                rgb_files = find_rgb_files(mosaic_dir)
                mask_files = find_mask_files(mosaic_dir)

                if len(rgb_files) == 0:
                    print(f"[WARNING] No RGB files found in {mosaic_dir / RGB_SUBDIR}")
                    continue

                if len(mask_files) == 0:
                    print(f"[WARNING] No masks found in {mosaic_dir / SEG_SUBDIR}")

                for rgb_file in rgb_files:
                    tile_number = extract_tile_number(rgb_file.name)
                    mask_file = match_mask(rgb_file, mask_files)

                    rgb = read_rgb(rgb_file)

                    if mask_file is not None:
                        mask = read_mask(mask_file)
                        overlay, mask = make_overlay(rgb, mask)
                    else:
                        mask = np.zeros(rgb.shape[:2], dtype=bool)
                        overlay = rgb.copy()
                        missing_masks.append(rgb_file)

                    tile_label = f"{tile_number:02d}" if tile_number is not None else "unknown"

                    title = (
                        f"{visit_name} | {mosaic_name} | Tile {tile_label}\n"
                        f"RGB: {rgb_file.name}"
                    )

                    if mask_file is not None:
                        title += f"\nMask: {mask_file.name}"
                    else:
                        title += "\nMask: NOT FOUND"

                    plot_tile_page(
                        pdf=pdf,
                        rgb=rgb,
                        mask=mask,
                        overlay=overlay,
                        title=title,
                    )

                    page_count += 1

        print(f"[DONE] {visit_name}: {page_count} pages")
        print(f"Saved PDF:\n{output_pdf}\n")

    if missing_masks:
        print(f"\nWARNING: {len(missing_masks)} RGB tiles had no matching mask.")
        for f in missing_masks[:30]:
            print(f"Missing mask for: {f}")
        if len(missing_masks) > 30:
            print("...")


if __name__ == "__main__":
    main()