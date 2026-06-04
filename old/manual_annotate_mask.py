#!/usr/bin/env python3

from pathlib import Path

import imageio.v3 as iio
import napari
import numpy as np
import tifffile as tiff


# ============================================================
# CONFIG
# ============================================================

rgb_path = Path(
    "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449/visit01/Mosaic03_4x4_FOV600_z110_32Sp/RGB/Im_00001_RGB.png"
).expanduser()

output_mask_path = Path(
    "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449/visit01/Mosaic03_4x4_FOV600_z110_32Sp/manual_mask/manual_mask.tif"
).expanduser()


# ============================================================
# IO
# ============================================================

def read_rgb_for_display(path: Path) -> np.ndarray:
    arr = iio.imread(path)
    arr = np.asarray(arr)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]

    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Invalid RGB image shape: {arr.shape}")

    arr = arr.astype(np.float32)

    if arr.max() > 1:
        arr = arr / 255.0

    return np.clip(arr, 0, 1)


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(str(path), mask.astype(np.uint16))
    print(f"[OK] Saved mask: {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    rgb = read_rgb_for_display(rgb_path)

    empty_mask = np.zeros(rgb.shape[:2], dtype=np.uint16)

    viewer = napari.Viewer(title="Manual segmentation on RGB")

    viewer.add_image(
        rgb,
        name="RGB",
        rgb=True,
    )

    labels_layer = viewer.add_labels(
        empty_mask,
        name="manual_mask",
        opacity=0.5,
    )

    print("\n[INFO] Napari opened.")
    print("[INFO] Draw manually on the 'manual_mask' layer.")
    print("[INFO] Use label value 1 for the mask.")
    print("[INFO] Press 'S' to save.")
    print("[INFO] Press 'Q' to save and close.\n")

    @viewer.bind_key("s")
    def save_current_mask(viewer):
        current_mask = np.asarray(labels_layer.data)
        current_mask = (current_mask > 0).astype(np.uint16)
        save_mask(output_mask_path, current_mask)

    @viewer.bind_key("q")
    def save_and_close(viewer):
        current_mask = np.asarray(labels_layer.data)
        current_mask = (current_mask > 0).astype(np.uint16)
        save_mask(output_mask_path, current_mask)
        viewer.close()

    napari.run()


if __name__ == "__main__":
    main()