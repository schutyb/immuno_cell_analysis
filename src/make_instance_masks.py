#!/usr/bin/env python3
"""
Create instance masks from SegData tile masks, filtering objects with equivalent diameter < 4 µm.
Pixel size: 0.5 µm/px  => min diameter 4 µm = 8 px.

It searches for folders named "SegData" under root and creates a sibling folder:
  .../SegData/           (input masks)
  .../instance_mask/     (output instance label masks + CSV tables)

Outputs per SegData folder:
  instance_mask/
    <tile_name>_inst.tif
    instances_<SegData_parent>.csv
    tiles_summary_<SegData_parent>.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from skimage.measure import label, regionprops_table
from skimage.segmentation import relabel_sequential

import tifffile as tiff
import imageio.v3 as iio


PIXEL_SIZE_UM = 0.5
MIN_DIAMETER_UM = 4.0
MIN_DIAMETER_PX = MIN_DIAMETER_UM / PIXEL_SIZE_UM  # 8 px


def read_mask(path: Path) -> np.ndarray:
    """Read an image mask (png/tif). Returns 2D array."""
    suffix = path.suffix.lower()
    if suffix in [".tif", ".tiff"]:
        arr = tiff.imread(str(path))
    else:
        arr = iio.imread(str(path))
    if arr.ndim == 3:
        # If RGB/ARGB, take first channel
        arr = arr[..., 0]
    return arr


def to_binary(mask: np.ndarray, fg_values: List[int] | None = None) -> np.ndarray:
    """
    Convert to boolean foreground mask.
    - If fg_values is None: foreground = mask > 0
    - Else: foreground = mask in fg_values
    """
    if fg_values is None:
        return mask > 0
    fg_values_np = np.array(fg_values, dtype=mask.dtype)
    return np.isin(mask, fg_values_np)


def build_instance_mask_and_props(
    bin_mask: np.ndarray,
    connectivity: int = 2
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Connected-components labeling + filter by equivalent diameter.
    Returns:
      inst_mask_filtered: 2D int32 (0 background, 1..N instances)
      props_df: dataframe per instance (including diameter_um and keep flag)
    """
    inst = label(bin_mask.astype(bool), connectivity=connectivity)

    if inst.max() == 0:
        empty_df = pd.DataFrame(columns=[
            "label", "area_px", "equivalent_diameter_px", "equivalent_diameter_um", "keep"
        ])
        return inst.astype(np.int32), empty_df

    props = regionprops_table(
        inst,
        properties=("label", "area", "equivalent_diameter")
    )
    props_df = pd.DataFrame(props).rename(columns={
        "area": "area_px",
        "equivalent_diameter": "equivalent_diameter_px",
    })
    props_df["equivalent_diameter_um"] = props_df["equivalent_diameter_px"] * PIXEL_SIZE_UM
    props_df["keep"] = props_df["equivalent_diameter_px"] >= MIN_DIAMETER_PX

    keep_labels = props_df.loc[props_df["keep"], "label"].to_numpy(dtype=np.int32)

    inst_filt = np.zeros_like(inst, dtype=np.int32)
    if keep_labels.size > 0:
        keep_mask = np.isin(inst, keep_labels)
        inst_filt = (inst * keep_mask).astype(np.int32)
        inst_filt, _, _ = relabel_sequential(inst_filt)

    return inst_filt, props_df


def save_instance_mask_tiff(out_path: Path, inst_mask: np.ndarray) -> None:
    """
    Save labeled instance mask as TIFF (lossless labels).
    Use uint16 if safe, else uint32.
    """
    max_label = int(inst_mask.max())
    if max_label <= np.iinfo(np.uint16).max:
        arr = inst_mask.astype(np.uint16)
    else:
        arr = inst_mask.astype(np.uint32)
    tiff.imwrite(str(out_path), arr, compression="deflate")


def find_segdata_dirs(root: Path) -> List[Path]:
    return [p for p in root.rglob("SegData") if p.is_dir() and p.name == "SegData"]


def process_segdata_dir(seg_dir: Path, fg_values: List[int] | None = None) -> None:
    """
    Process one SegData folder. Creates sibling instance_mask folder and writes outputs.
    """
    out_dir = seg_dir.parent / "instance_mask"
    out_dir.mkdir(parents=True, exist_ok=True)

    img_paths: List[Path] = []
    for ext in ("*.png", "*.tif", "*.tiff"):
        img_paths.extend(sorted(seg_dir.glob(ext)))

    if len(img_paths) == 0:
        print(f"[WARN] No masks found in {seg_dir}")
        return

    all_instances_rows = []
    tile_summary_rows = []

    for img_path in img_paths:
        mask = read_mask(img_path)
        bin_mask = to_binary(mask, fg_values=fg_values)

        inst_mask, props_df = build_instance_mask_and_props(bin_mask, connectivity=2)

        # Save instance mask
        out_name = img_path.stem + "_inst.tif"
        save_instance_mask_tiff(out_dir / out_name, inst_mask)

        # Add tile info to per-instance table
        if len(props_df) > 0:
            props_df = props_df.copy()
            props_df.insert(0, "tile_file", img_path.name)
            props_df.insert(1, "tile_stem", img_path.stem)
            all_instances_rows.append(props_df)

        # Tile summary
        n_before = int(label(bin_mask, connectivity=2).max())
        n_after = int(inst_mask.max())
        tile_summary_rows.append({
            "tile_file": img_path.name,
            "tile_stem": img_path.stem,
            "n_objects_before": n_before,
            "n_instances_after": n_after,
            "min_diameter_um": MIN_DIAMETER_UM,
            "pixel_size_um": PIXEL_SIZE_UM,
        })

    parent_name = seg_dir.parent.name  # e.g., Mosaic03_... depending on your structure

    if all_instances_rows:
        instances_df = pd.concat(all_instances_rows, ignore_index=True)
    else:
        instances_df = pd.DataFrame(columns=[
            "tile_file", "tile_stem", "label", "area_px", "equivalent_diameter_px",
            "equivalent_diameter_um", "keep"
        ])

    tiles_df = pd.DataFrame(tile_summary_rows)

    instances_csv = out_dir / f"instances_{parent_name}.csv"
    tiles_csv = out_dir / f"tiles_summary_{parent_name}.csv"

    instances_df.to_csv(instances_csv, index=False)
    tiles_df.to_csv(tiles_csv, index=False)

    print(f"[OK] {seg_dir} -> {out_dir} | tiles={len(img_paths)}")


def main():
    # ✅ Put your default data root path here:
    DEFAULT_ROOT = Path("/Users/schutyb/Documents/balu_lab/dod/data_raw/patients").expanduser()

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default=str(DEFAULT_ROOT),
        help=f"Root folder containing all patients/visits/mosaics (default: {DEFAULT_ROOT})",
    )
    ap.add_argument(
        "--fg-values",
        type=str,
        default="",
        help="Optional: comma-separated list of integer pixel values to treat as foreground (e.g. '255' or '1,2'). "
             "If empty, uses mask > 0.",
    )
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root path not found: {root}")

    fg_values = None
    if args.fg_values.strip():
        fg_values = [int(x.strip()) for x in args.fg_values.split(",") if x.strip()]

    seg_dirs = find_segdata_dirs(root)
    if not seg_dirs:
        print(f"[WARN] No SegData folders found under {root}")
        return

    print(f"Found {len(seg_dirs)} SegData folders under {root}")
    for seg_dir in seg_dirs:
        process_segdata_dir(seg_dir, fg_values=fg_values)


if __name__ == "__main__":
    main()