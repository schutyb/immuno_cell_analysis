"""
mask_analysis.py
================

Goal
----
Build a full mosaic mask from 16 per-tile segmentation masks, convert it into an
INSTANCE mask (one integer label per object), remove tiny objects by minimum
equivalent diameter, and compute per-object morphology features saved to CSV.

This script is part of the melanoma FLIM pipeline:
- Raw FLIM mosaics are 4x4 tiles (16 tiles) assembled with a known serpentine scan order.
- Each Mosaic folder contains a mask folder: "mask_*"
  which contains 16 mask images (often PNG), one per tile.
- We assemble the 16 masks into a full 4x4 mosaic mask with the same geometry as the FLIM mosaic.

Key steps
---------
For each Mosaic folder:
1) Locate mask folder: the unique subfolder starting with "mask_" (newest if multiple)
2) Gather the first 16 mask image files in ALPHABETICAL order:
      tile 1 -> first file
      tile 2 -> second file
      ...
      tile 16 -> 16th file
   (ASSUMPTION: your student masks are consistently saved alphabetically in the correct tile order)
3) Load each mask tile (PNG/JPG/TIF supported), convert to 2D, then assemble 4x4 mosaic
   using the serpentine scan order.
4) Convert mosaic to binary mask: (mask > BIN_THRESH)
5) Connected-component labeling -> instance mask
6) Compute regionprops features per object
7) Filter objects by equivalent diameter >= MIN_EQ_DIAMETER_PX
8) Relabel sequentially (1..N) after filtering
9) Save:
   - Optional debug binary mosaic TIFF: mask_mosaic_binary.tif  (0/255)
   - Final instance mask TIFF: mask_instances_minEqDiam{d}px.tif (uint32 labels)
   - Features CSV: mask_instances_features_minEqDiam{d}px.csv

Folder structure expected
-------------------------
PATIENT_DIR/
  visit_01/
    Mosaic.../
      mask_Mosaic.../    <-- this script reads from here
        <16 mask image tiles: png/jpg/tif>
  visit_02/
    ...

Outputs are written inside each mask folder.

Outputs
-------
Inside each mask folder (mask_*):
- mask_mosaic_binary.tif                            (optional debug)
- mask_instances_minEqDiam8px.tif                   (uint32 instance labels)
- mask_instances_features_minEqDiam8px.csv          (morphology features per instance)

CSV columns (English)
---------------------
label
area_px2
equivalent_diameter_px
perimeter_px
circularity
eccentricity
major_axis_length_px
minor_axis_length_px
major_minor_ratio
centroid_y
centroid_x

Important assumptions / gotchas
-------------------------------
- Mask tiles are NOT necessarily TIFF; often PNG. We support png/jpg/tif.
- Mask tiles are assumed to be alphabetically ordered and correspond to tiles 1..16.
- Mosaic assembly uses TILE_ORDER_4x4 serpentine order (must match FLIM mosaics).
- Equivalent diameter filtering is applied AFTER instance labeling (per object).
- The debug binary mosaic is the FULL binary mask without diameter filtering.

How to run
----------
Edit PATIENT_DIR (top CONFIG) then:
    python mask_analysis.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import tifffile as tiff
import pandas as pd
import imageio.v3 as iio

from skimage.measure import label, regionprops
from skimage.segmentation import relabel_sequential


# ============================================================
# CONFIG (edit these parameters)
# ============================================================
PATIENT_DIR = Path("/Users/schutyb/Documents/balu_lab/data_patient_449")

MASK_PREFIX = "mask_"   # folder inside each Mosaic*
BIN_THRESH = 0          # binary = (mask > BIN_THRESH)

MIN_EQ_DIAMETER_PX = 8.0
CONNECTIVITY = 2        # 1=4-neigh, 2=8-neigh

# Save a debug binary mosaic TIFF (full mask, BEFORE diameter filtering)
SAVE_BINARY_DEBUG = True

# Mosaic tile order (same as raw FLIM mosaic)
TILE_ORDER_4x4 = np.array(
    [
        [1, 2, 3, 4],
        [8, 7, 6, 5],
        [9, 10, 11, 12],
        [16, 15, 14, 13],
    ],
    dtype=int,
)

OUT_BINARY_NAME = "mask_mosaic_binary.tif"
OUT_INST_NAME = "mask_instances_minEqDiam{d}px.tif"
OUT_CSV_NAME = "mask_instances_features_minEqDiam{d}px.csv"


# ============================================================
# Folder helpers
# ============================================================
def list_visits(patient_dir: Path) -> list[Path]:
    """Return all visit_* folders in PATIENT_DIR."""
    return sorted([p for p in patient_dir.iterdir() if p.is_dir() and p.name.startswith("visit_")])


def list_mosaics(visit_dir: Path) -> list[Path]:
    """Return all Mosaic* folders inside a visit folder."""
    return sorted([p for p in visit_dir.iterdir() if p.is_dir() and p.name.startswith("Mosaic")])


def find_mask_folder(mosaic_dir: Path) -> Path | None:
    """
    Find mask folder inside a Mosaic folder.

    If multiple mask_* folders exist, chooses the newest by modified time.
    """
    matches = sorted([p for p in mosaic_dir.iterdir() if p.is_dir() and p.name.startswith(MASK_PREFIX)])
    if not matches:
        return None
    if len(matches) > 1:
        matches = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


# ============================================================
# Load mask tiles + assemble mosaic
# ============================================================
def load_mask_2d(path: Path) -> np.ndarray:
    """
    Load a single mask tile (PNG/JPG/TIF) into a 2D numpy array.

    - For TIFF: uses tifffile.imread
    - For others: uses imageio.v3.imread
    - If a 3-channel image is read (RGB/RGBA), we use the first channel.
    """
    ext = path.suffix.lower()
    if ext in (".tif", ".tiff"):
        arr = tiff.imread(str(path))
    else:
        arr = iio.imread(str(path))

    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.ndim == 3:
        # (H,W,C) or (C,H,W): take first channel or max as fallback
        if arr.shape[-1] in (3, 4):
            arr = arr[..., 0]
        elif arr.shape[0] in (3, 4):
            arr = arr[0, ...]
        else:
            arr = arr.max(axis=-1)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask tile. Got {arr.shape} for {path}")

    return arr


def gather_mask_tiles_alphabetical(mask_dir: Path) -> dict[int, Path]:
    """
    Gather the first 16 mask files, sorted alphabetically by filename.

    ASSUMPTION:
    - Mask tiles are always alphabetically ordered and correspond to tiles 1..16.

    Returns
    -------
    dict: {1: path_to_tile1, ..., 16: path_to_tile16}
    """
    exts_ok = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    files = sorted(
        [p for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() in exts_ok],
        key=lambda p: p.name,
    )

    if len(files) < 16:
        raise FileNotFoundError(f"Expected >=16 mask images in {mask_dir}, found {len(files)}")

    files16 = files[:16]
    return {i + 1: files16[i] for i in range(16)}  # 1..16


def assemble_mosaic_4x4(tiles: dict[int, np.ndarray]) -> np.ndarray:
    """
    Assemble 16 2D tiles into a single 4x4 mosaic using serpentine TILE_ORDER_4x4.
    """
    missing = [i for i in range(1, 17) if i not in tiles]
    if missing:
        raise FileNotFoundError(f"Missing mask tiles: {missing}")

    shapes = {tiles[i].shape for i in range(1, 17)}
    if len(shapes) != 1:
        raise ValueError(f"Mask tile shapes differ: {shapes}")

    h, w = next(iter(shapes))
    mosaic = np.zeros((h * 4, w * 4), dtype=tiles[1].dtype)

    for r in range(4):
        for c in range(4):
            idx = int(TILE_ORDER_4x4[r, c])
            y0, y1 = r * h, (r + 1) * h
            x0, x1 = c * w, (c + 1) * w
            mosaic[y0:y1, x0:x1] = tiles[idx]

    return mosaic


def build_mask_mosaic_binary(mask_dir: Path) -> np.ndarray:
    """
    Build full mosaic binary mask from a mask_* folder.

    Returns
    -------
    binary : np.ndarray of shape (Y, X), dtype=bool
    """
    tile_paths = gather_mask_tiles_alphabetical(mask_dir)

    tiles = {}
    for i in range(1, 17):
        tiles[i] = load_mask_2d(tile_paths[i])

    mosaic = assemble_mosaic_4x4(tiles)
    return mosaic > BIN_THRESH


# ============================================================
# Instances + features + filtering
# ============================================================
def circularity(area: float, perimeter: float) -> float:
    """Circularity = 4πA / P²."""
    if perimeter <= 0:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter ** 2))


def map_label(fw, x: int) -> int:
    """
    Robust mapping function for relabel_sequential forward_map output.
    Supports different fw types (callable, dict-like, array-like).
    """
    if callable(fw):
        return int(fw(x))
    if hasattr(fw, "get"):
        return int(fw.get(x, 0))
    try:
        return int(fw[x])
    except Exception:
        return 0


def instances_and_features(binary: np.ndarray, min_eq_diameter_px: float, connectivity: int):
    """
    Convert a binary mask into:
    - an instance label mask (uint32)
    - a DataFrame of per-instance morphological features

    Filtering:
    - Objects with equivalent diameter < min_eq_diameter_px are removed
    - Remaining objects are relabeled sequentially 1..N

    Returns
    -------
    inst_final : np.ndarray uint32 (Y,X)
    df : pd.DataFrame
    """
    inst = label(binary.astype(bool), connectivity=connectivity)
    if inst.max() == 0:
        return inst.astype(np.uint32), pd.DataFrame()

    props = regionprops(inst)

    keep_old = []
    rows = []

    for p in props:
        area = float(p.area)
        perimeter = float(p.perimeter)
        eq_diameter = float(p.equivalent_diameter)
        major_axis = float(p.major_axis_length)
        minor_axis = float(p.minor_axis_length)
        eccentricity = float(p.eccentricity)
        cy, cx = float(p.centroid[0]), float(p.centroid[1])

        # Filter by equivalent diameter
        if eq_diameter < min_eq_diameter_px:
            continue

        keep_old.append(int(p.label))
        rows.append(
            {
                "old_label": int(p.label),
                "area_px2": area,
                "equivalent_diameter_px": eq_diameter,
                "perimeter_px": perimeter,
                "circularity": circularity(area, perimeter),
                "eccentricity": eccentricity,
                "major_axis_length_px": major_axis,
                "minor_axis_length_px": minor_axis,
                "major_minor_ratio": (major_axis / minor_axis) if minor_axis > 0 else np.nan,
                "centroid_y": cy,
                "centroid_x": cx,
            }
        )

    # If everything filtered out, return empty instance mask + empty df
    if not keep_old:
        return np.zeros_like(inst, dtype=np.uint32), pd.DataFrame()

    keep_old = np.asarray(keep_old, dtype=np.int32)
    inst_kept = np.where(np.isin(inst, keep_old), inst, 0).astype(np.int32)

    # Relabel sequentially to 1..N
    inst_final, fw, _ = relabel_sequential(inst_kept)

    df = pd.DataFrame(rows)
    df["label"] = df["old_label"].map(lambda x: map_label(fw, int(x)))
    df = df.drop(columns=["old_label"]).sort_values("label").reset_index(drop=True)

    return inst_final.astype(np.uint32), df


# ============================================================
# Save helpers
# ============================================================
def save_binary_tiff(path: Path, binary: np.ndarray):
    """
    Save binary mask as a TIFF (0/255) for easy viewing in ImageJ / napari.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(
        str(path),
        (binary.astype(np.uint8) * 255),
        photometric="minisblack",
        metadata={"axes": "YX"},
    )


def save_instance_tiff(path: Path, inst: np.ndarray):
    """
    Save instance mask as uint32 TIFF. Label 0 is background, 1..N are objects.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(
        str(path),
        inst.astype(np.uint32),
        photometric="minisblack",
        metadata={"axes": "YX"},
    )


# ============================================================
# Main
# ============================================================
def main():
    visits = list_visits(PATIENT_DIR)
    if not visits:
        raise FileNotFoundError(f"No visit_* in: {PATIENT_DIR}")

    print(f"[INFO] Root: {PATIENT_DIR}")
    print(f"[INFO] Found {len(visits)} visit(s)")

    for visit_dir in visits:
        mosaics = list_mosaics(visit_dir)
        print(f"\n[VISIT] {visit_dir.name} | mosaics: {len(mosaics)}")

        for mosaic_dir in mosaics:
            mask_dir = find_mask_folder(mosaic_dir)
            if mask_dir is None:
                print(f"  [WARN] {mosaic_dir.name}: no {MASK_PREFIX}* folder")
                continue

            print(f"  [MOSAIC] {mosaic_dir.name}")
            print(f"    mask_dir: {mask_dir.name}")

            binary = build_mask_mosaic_binary(mask_dir)
            print(f"    binary mosaic shape: {binary.shape} | fg px: {int(binary.sum())}")

            inst, df = instances_and_features(binary, MIN_EQ_DIAMETER_PX, CONNECTIVITY)

            # Debug binary mosaic (full mask without size filtering)
            if SAVE_BINARY_DEBUG:
                save_binary_tiff(mask_dir / OUT_BINARY_NAME, binary)

            # Final instance mask + features table (after min diameter filtering)
            out_inst = mask_dir / OUT_INST_NAME.format(d=int(MIN_EQ_DIAMETER_PX))
            out_csv = mask_dir / OUT_CSV_NAME.format(d=int(MIN_EQ_DIAMETER_PX))

            save_instance_tiff(out_inst, inst)
            df.to_csv(out_csv, index=False)

            print(f"    saved: {out_inst.name} | instances kept: {int(inst.max())}")
            print(f"    saved: {out_csv.name}  | rows: {len(df)}")


if __name__ == "__main__":
    main()