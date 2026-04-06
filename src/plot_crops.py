#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import math

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# CONFIG
# ============================================================

PATIENT_DIR = Path("/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449")
CROPS_ROOT = PATIENT_DIR / "analysis" / "cell_families_lifetime" / "crops"

OUTPUT_DIR = PATIENT_DIR / "analysis" / "cell_families_lifetime" / "montages"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VISITS = ["visit01", "visit02", "visit03", "visit04"]
FAMILIES = ["family_0", "family_1", "family_2"]

N_CROPS_PER_ROW = 12

# visual layout
CELL_W = 45
CELL_H = 45
PAD_X = 6
PAD_Y = 10
LEFT_LABEL_W = 90
TOP_TITLE_H = 40
BG_COLOR = (255, 255, 255)
TEXT_COLOR = (0, 0, 0)
EMPTY_COLOR = (240, 240, 240)


# ============================================================
# HELPERS
# ============================================================

def short_visit_label(visit: str) -> str:
    visit = visit.lower()
    if visit.startswith("visit"):
        try:
            return f"Visit {int(visit.replace('visit', ''))}"
        except ValueError:
            return visit
    return visit


def load_crop(path: Path, size=(CELL_W, CELL_H)) -> Image.Image:
    arr = iio.imread(path)

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    img = Image.fromarray(arr)
    if img.size != size:
        img = img.resize(size, Image.Resampling.NEAREST)
    return img


def blank_cell(size=(CELL_W, CELL_H)) -> Image.Image:
    arr = np.full((size[1], size[0], 3), EMPTY_COLOR, dtype=np.uint8)
    return Image.fromarray(arr)


# ============================================================
# MAIN
# ============================================================

def main():
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
        font_small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    for family in FAMILIES:
        # canvas size
        width = LEFT_LABEL_W + N_CROPS_PER_ROW * CELL_W + (N_CROPS_PER_ROW - 1) * PAD_X + 20
        height = TOP_TITLE_H + len(VISITS) * CELL_H + (len(VISITS) - 1) * PAD_Y + 20

        canvas = Image.new("RGB", (width, height), BG_COLOR)
        draw = ImageDraw.Draw(canvas)

        # title
        title = f"{family.replace('_', ' ').title()} - first {N_CROPS_PER_ROW} cells per visit"
        draw.text((10, 10), title, fill=TEXT_COLOR, font=font)

        for row_idx, visit in enumerate(VISITS):
            y = TOP_TITLE_H + row_idx * (CELL_H + PAD_Y)

            # row label
            draw.text((10, y + CELL_H // 3), short_visit_label(visit), fill=TEXT_COLOR, font=font_small)

            family_dir = CROPS_ROOT / visit / family
            crop_paths = []

            if family_dir.exists():
                crop_paths = sorted(
                    [p for p in family_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}]
                )[:N_CROPS_PER_ROW]

            for col_idx in range(N_CROPS_PER_ROW):
                x = LEFT_LABEL_W + col_idx * (CELL_W + PAD_X)

                if col_idx < len(crop_paths):
                    img = load_crop(crop_paths[col_idx], size=(CELL_W, CELL_H))
                else:
                    img = blank_cell(size=(CELL_W, CELL_H))

                canvas.paste(img, (x, y))

        out_path = OUTPUT_DIR / f"{family}_montage_by_visit.png"
        canvas.save(out_path)
        print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()