#!/usr/bin/env python3

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import tifffile as tiff


# ----------------------------
# CONFIG
# ----------------------------

PSEUDOCOLOR_PATH = Path(
"/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449/visit01/Mosaic03_4x4_FOV600_z110_32Sp/_new/pseudocolor_phase_first_harmonic_upsampled.png"
).expanduser()

MASK_PATH = Path(
"/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449/visit01/Mosaic03_4x4_FOV600_z110_32Sp/_new/instance_mask_filtered.tif"
).expanduser()

ROI_X = 0
ROI_Y = 0
ROI_WIDTH = 4800
ROI_HEIGHT = 4800

GIF_DURATION = 1


# ----------------------------
# IO
# ----------------------------

def read_image(path):

    if path.suffix.lower() in [".tif", ".tiff"]:
        arr = tiff.imread(str(path))
    else:
        arr = iio.imread(str(path))

    arr = np.asarray(arr)

    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]

    if arr.max() > 1:
        arr = arr / 255.0

    return arr.astype(np.float32)


def save_png(path, img):

    img = np.clip(img, 0, 1)
    iio.imwrite(str(path), (255 * img).astype(np.uint8))


# ----------------------------
# ROI crop
# ----------------------------

def crop(arr):

    return arr[
        ROI_Y : ROI_Y + ROI_HEIGHT,
        ROI_X : ROI_X + ROI_WIDTH,
    ]


# ----------------------------
# Overlay mask (ROI blanca completa)
# ----------------------------

def overlay_full_roi(rgb, mask):

    mask_bool = mask > 0

    out = rgb.copy()

    # pintar TODA la ROI blanca
    out[mask_bool] = [1.0, 1.0, 1.0]

    return out


# ----------------------------
# GIF
# ----------------------------

def save_gif(path, img1, img2):

    frames = [
        (255 * img1).astype(np.uint8),
        (255 * img2).astype(np.uint8),
    ]

    iio.imwrite(
        str(path),
        frames,
        duration=GIF_DURATION,
        loop=0,
    )


# ----------------------------
# MAIN
# ----------------------------

def main():

    pseudo = read_image(PSEUDOCOLOR_PATH)
    mask = read_image(MASK_PATH)

    if mask.ndim == 3:
        mask = mask[:, :, 0]

    pseudo_crop = crop(pseudo)
    mask_crop = crop(mask)

    overlay = overlay_full_roi(pseudo_crop, mask_crop)

    out_dir = PSEUDOCOLOR_PATH.parent / "roi_overlay"
    out_dir.mkdir(exist_ok=True)

    save_png(out_dir / "roi_pseudocolor.png", pseudo_crop)
    save_png(out_dir / "roi_overlay_white.png", overlay)

    save_gif(
        out_dir / "roi_overlay.gif",
        pseudo_crop,
        overlay,
    )

    print("[OK] Saved results in:", out_dir)


if __name__ == "__main__":
    main()