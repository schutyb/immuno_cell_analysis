#!/usr/bin/env python3

from pathlib import Path
import re

import imageio.v3 as iio
import napari
import numpy as np
import tifffile as tiff
from scipy.ndimage import binary_erosion


# ============================================================
# CONFIG
# ============================================================

input_tiles = Path(
    "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449/visit01/Mosaic03_4x4_FOV600_z110_32Sp/SegData"
).expanduser()

color_image_path = Path(
    "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449/visit01/Mosaic03_4x4_FOV600_z110_32Sp/_new/pseudocolor_phase_first_harmonic_upsampled.png"
).expanduser()

output_mask_path = Path(
    "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449/visit01/Mosaic03_4x4_FOV600_z110_32Sp/_new/new_mask/new_mask.tif"
).expanduser()

grid_rows = 4
grid_cols = 4


# ============================================================
# IO
# ============================================================

def read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() in [".tif", ".tiff"]:
        arr = tiff.imread(str(path))
    else:
        arr = iio.imread(str(path))

    arr = np.asarray(arr)

    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]

    return arr


def read_rgb_for_display(path: Path) -> np.ndarray:
    arr = read_image(path).astype(np.float32)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    if arr.ndim != 3:
        raise ValueError(f"Invalid pseudocolor shape: {arr.shape}")

    if arr.shape[-1] == 4:
        arr = arr[..., :3]

    if arr.max() > 1:
        arr = arr / 255.0

    return np.clip(arr, 0, 1)


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(str(path), mask.astype(np.uint16))
    print(f"[OK] Saved mask: {path}")


# ============================================================
# SORTING
# ============================================================

def natural_key(text: str):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", text)]


def list_tile_paths(folder: Path):
    valid_ext = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    paths = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in valid_ext]
    return sorted(paths, key=lambda p: natural_key(p.name))


# ============================================================
# SNAKE MOSAIC
# ============================================================

def build_snake_indices(rows: int, cols: int):
    order = []
    idx = 0
    for r in range(rows):
        row_ids = list(range(idx, idx + cols))
        if r % 2 == 1:
            row_ids = row_ids[::-1]
        order.extend(row_ids)
        idx += cols
    return order


def build_mosaic_from_tiles_snake(tile_paths, rows=4, cols=4):
    expected = rows * cols
    if len(tile_paths) != expected:
        raise ValueError(f"Expected {expected} tiles, found {len(tile_paths)}")

    tiles = []
    for p in tile_paths:
        arr = read_image(p)

        if arr.ndim == 3:
            arr = arr[..., 0]

        if arr.ndim != 2:
            raise ValueError(f"Tile {p.name} has invalid shape {arr.shape}")

        tiles.append(arr)

    tile_h, tile_w = tiles[0].shape

    for i, arr in enumerate(tiles):
        if arr.shape != (tile_h, tile_w):
            raise ValueError(
                f"Shape mismatch in {tile_paths[i].name}: {arr.shape} != {(tile_h, tile_w)}"
            )

    mosaic = np.zeros((rows * tile_h, cols * tile_w), dtype=tiles[0].dtype)

    snake_order = build_snake_indices(rows, cols)

    for visual_pos, tile_idx in enumerate(snake_order):
        r = visual_pos // cols
        c = visual_pos % cols

        y0 = r * tile_h
        y1 = (r + 1) * tile_h
        x0 = c * tile_w
        x1 = (c + 1) * tile_w

        mosaic[y0:y1, x0:x1] = tiles[tile_idx]

    return mosaic


# ============================================================
# BORDES
# ============================================================

def mask_to_outline(mask: np.ndarray) -> np.ndarray:
    mask_bool = mask > 0
    eroded = binary_erosion(mask_bool)
    outline = mask_bool & (~eroded)
    return outline.astype(np.uint8)


# ============================================================
# MAIN
# ============================================================

def main():
    tile_paths = list_tile_paths(input_tiles)

    print("[INFO] Found tiles:")
    for p in tile_paths:
        print("   ", p.name)

    mask_mosaic = build_mosaic_from_tiles_snake(
        tile_paths,
        rows=grid_rows,
        cols=grid_cols,
    )

    color_img = read_rgb_for_display(color_image_path)

    if color_img.shape[:2] != mask_mosaic.shape:
        raise ValueError(
            f"Size mismatch:\n"
            f"  pseudocolor = {color_img.shape[:2]}\n"
            f"  mask mosaic = {mask_mosaic.shape}\n"
            f"Both must have the same XY size."
        )

    editable_mask = (mask_mosaic > 0).astype(np.uint16)
    outline_mask = mask_to_outline(editable_mask)

    viewer = napari.Viewer(title="Mask editor on pseudocolor")

    viewer.add_image(
        color_img,
        name="pseudocolor",
        rgb=True,
    )

    # capa editable real
    labels_layer = viewer.add_labels(
        editable_mask,
        name="editable_mask",
        opacity=0.0,
        visible=False,
    )

    # capa visual solo de contornos blancos
    outline_layer = viewer.add_image(
        outline_mask,
        name="mask_outline",
        colormap="gray",
        opacity=1.0,
    )

    print("\n[INFO] Napari opened.")
    print("[INFO] Edit the layer 'editable_mask'.")
    print("[INFO] If needed, make 'editable_mask' visible from the layer list.")
    print("[INFO] The visible overlay shows only white borders.")
    print("[INFO] Press 'U' to refresh borders after editing.")
    print("[INFO] Press 'S' to save.")
    print("[INFO] Press 'Q' to save and close.\n")

    @viewer.bind_key("u")
    def update_outline(viewer):
        current_mask = np.asarray(labels_layer.data)
        current_mask = (current_mask > 0).astype(np.uint16)
        outline_layer.data = mask_to_outline(current_mask)
        print("[OK] Outline updated")

    @viewer.bind_key("s")
    def save_current_mask(viewer):
        current_mask = np.asarray(labels_layer.data)
        current_mask = (current_mask > 0).astype(np.uint16)
        outline_layer.data = mask_to_outline(current_mask)
        save_mask(output_mask_path, current_mask)

    @viewer.bind_key("q")
    def save_and_close(viewer):
        current_mask = np.asarray(labels_layer.data)
        current_mask = (current_mask > 0).astype(np.uint16)
        outline_layer.data = mask_to_outline(current_mask)
        save_mask(output_mask_path, current_mask)
        viewer.close()

    napari.run()


if __name__ == "__main__":
    main()