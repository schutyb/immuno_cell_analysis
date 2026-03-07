#!/usr/bin/env python3

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import imageio.v3 as iio
import numpy as np
import pandas as pd
import tifffile as tiff
from phasorpy.lifetime import phasor_calibrate
from phasorpy.phasor import phasor_from_signal
from skimage.measure import label, regionprops_table
from skimage.segmentation import relabel_sequential


# ----------------------------
# User settings
# ----------------------------
FREQUENCY_MHZ = 80.0
COUMARIN_LIFETIME_NS = 2.5

PIXEL_SIZE_UM = 0.5
MIN_DIAMETER_UM = 4.0
MIN_DIAMETER_PX = MIN_DIAMETER_UM / PIXEL_SIZE_UM  # 8 px

CONNECTIVITY = 1  # better for discrete objects like cells


# ----------------------------
# Basic IO
# ----------------------------
def read_image(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        arr = tiff.imread(str(path))
    else:
        arr = iio.imread(str(path))

    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr


def save_tiff(path: Path, arr: np.ndarray) -> None:
    tiff.imwrite(str(path), arr, compression="deflate")


def save_png(path: Path, arr: np.ndarray) -> None:
    x = np.asarray(arr, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if x.ndim == 2:
        if np.max(x) > np.min(x):
            x = (x - np.min(x)) / (np.max(x) - np.min(x))
        else:
            x = np.zeros_like(x, dtype=np.float32)
        iio.imwrite(str(path), (255 * x).astype(np.uint8))
        return

    if x.ndim == 3:
        x = np.clip(x, 0.0, 1.0)
        iio.imwrite(str(path), (255 * x).astype(np.uint8))
        return

    raise ValueError(f"Unsupported ndim={x.ndim} for PNG output")


# ----------------------------
# Folder discovery
# ----------------------------
def find_coumarin_dir(visit_dir: Path) -> Path:
    dirs = [p for p in visit_dir.iterdir() if p.is_dir() and p.name.lower().startswith("coumarin_")]
    if len(dirs) == 0:
        raise FileNotFoundError(f"No coumarin_* folder found in {visit_dir}")
    if len(dirs) > 1:
        print(f"[WARN] Multiple coumarin folders found, using first: {dirs[0].name}")
    return sorted(dirs)[0]


def find_mosaic_dirs(visit_dir: Path) -> List[Path]:
    out = []
    for p in visit_dir.iterdir():
        if not p.is_dir():
            continue
        if p.name.lower().startswith("coumarin_"):
            continue
        if (p / "flim").exists() and (p / "SegData").exists():
            out.append(p)
    return sorted(out)


# ----------------------------
# Tile indexing / snake order
# ----------------------------
def infer_grid(n_tiles: int) -> Tuple[int, int]:
    side = int(round(math.sqrt(n_tiles)))
    if side * side != n_tiles:
        raise ValueError(f"Tile count {n_tiles} is not a perfect square")
    return side, side


def snake_position(tile_id: int, ncols: int) -> Tuple[int, int]:
    """
    tile_id is 1-based
    """
    idx = tile_id - 1
    row = idx // ncols
    pos = idx % ncols
    col = pos if row % 2 == 0 else (ncols - 1 - pos)
    return row, col


def tile_id_from_flim_name(name: str) -> int | None:
    """
    Im_00001.tif -> 1
    """
    m = re.search(r"im_(\d+)", name.lower())
    if m:
        return int(m.group(1))
    return None


def tile_id_from_mask_name(name: str) -> int | None:
    """
    ..._t01.png, ..._t16_cell_inst.tif, ...
    """
    name = name.lower()
    patterns = [
        r"_t(\d{1,2})(?=_|$)",
        r"t(\d{1,2})(?=_|$)",
    ]
    for pat in patterns:
        m = re.search(pat, name)
        if m:
            return int(m.group(1))
    return None


def list_flim_tiles(folder: Path) -> Dict[int, Path]:
    files = list(folder.glob("Im_*.tif")) + list(folder.glob("Im_*.tiff"))
    out: Dict[int, Path] = {}
    for f in sorted(files):
        tid = tile_id_from_flim_name(f.stem)
        if tid is not None:
            out[tid] = f
    return out


def list_mask_tiles(folder: Path) -> Dict[int, Path]:
    files: List[Path] = []
    for ext in ("*.png", "*.tif", "*.tiff", "*.jpg", "*.jpeg"):
        files.extend(folder.glob(ext))

    out: Dict[int, Path] = {}
    for f in sorted(files):
        tid = tile_id_from_mask_name(f.stem)
        if tid is not None:
            out[tid] = f
    return out


# ----------------------------
# FLIM signal handling
# ----------------------------
def mode_from_folder_name(name: str) -> str:
    lower = name.lower()
    if "32sp" in lower:
        return "32sp"
    if "32a1" in lower:
        return "32a1"
    return "unknown"


def move_hist_axis_first(arr: np.ndarray) -> np.ndarray:
    """
    Try to move TCSPC histogram axis to axis 0.
    Expected output shape: (bins, y, x)
    """
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.ndim == 3:
        if arr.shape[0] <= 256:
            return arr
        hist_axis = int(np.argmin(arr.shape))
        return np.moveaxis(arr, hist_axis, 0)

    if arr.ndim == 2:
        raise ValueError(f"Signal looks 2D only, expected 3D histogram stack. shape={arr.shape}")

    if arr.ndim > 3:
        hist_axis = int(np.argmin(arr.shape))
        arr = np.moveaxis(arr, hist_axis, 0)
        if arr.ndim != 3:
            raise ValueError(f"Unsupported signal shape after moving histogram axis: {arr.shape}")
        return arr

    raise ValueError(f"Unsupported signal shape: {arr.shape}")


def crop_signal_bins(signal: np.ndarray, mode: str) -> np.ndarray:
    """
    32Sp -> use first 16 bins
    32A1 -> use full histogram
    """
    if mode == "32sp":
        if signal.shape[0] < 16:
            raise ValueError(f"32Sp mode but histogram has only {signal.shape[0]} bins")
        return signal[:16]
    return signal


def load_flim_signal(path: Path, mode: str) -> np.ndarray:
    arr = read_image(path)
    arr = move_hist_axis_first(arr)
    arr = crop_signal_bins(arr, mode)
    return arr.astype(np.float32)


# ----------------------------
# Calibration reference
# ----------------------------
def find_single_tif(folder: Path) -> Path:
    files = list(folder.glob("*.tif")) + list(folder.glob("*.tiff"))
    if len(files) == 0:
        raise FileNotFoundError(f"No TIFF found in {folder}")
    if len(files) > 1:
        print(f"[WARN] Multiple TIFFs in {folder}, using first: {files[0].name}")
    return sorted(files)[0]


def build_reference_phasor(coumarin_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    coumarin_tif = find_single_tif(coumarin_dir)
    mode = mode_from_folder_name(coumarin_dir.name)
    signal = load_flim_signal(coumarin_tif, mode=mode)

    mean_ref, real_ref, imag_ref = phasor_from_signal(signal, harmonic=[1, 2], axis=0)

    return (
        mean_ref.astype(np.float32),
        np.asarray(real_ref, dtype=np.float32),
        np.asarray(imag_ref, dtype=np.float32),
    )


# ----------------------------
# SegData -> labeled mask
# ----------------------------
def is_binary_mask(mask: np.ndarray) -> bool:
    unique_vals = np.unique(mask)
    unique_nonzero = unique_vals[unique_vals > 0]
    return len(unique_nonzero) <= 1


def build_labeled_mask(mask: np.ndarray) -> np.ndarray:
    """
    If mask already contains multiple labels, preserve them.
    If binary, create connected components.
    """
    mask = np.asarray(mask)

    if is_binary_mask(mask):
        return label(mask > 0, connectivity=CONNECTIVITY).astype(np.int32)

    labeled = mask.astype(np.int32).copy()
    labeled[labeled < 0] = 0
    labeled, _, _ = relabel_sequential(labeled)
    return labeled.astype(np.int32)


def filter_labels_by_diameter(inst: np.ndarray) -> Tuple[np.ndarray, pd.DataFrame]:
    if inst.max() == 0:
        empty = pd.DataFrame(columns=[
            "label", "area_px", "equivalent_diameter_px", "equivalent_diameter_um", "keep"
        ])
        return inst.astype(np.int32), empty

    props = regionprops_table(inst, properties=("label", "area", "equivalent_diameter"))
    df = pd.DataFrame(props).rename(columns={
        "area": "area_px",
        "equivalent_diameter": "equivalent_diameter_px",
    })
    df["equivalent_diameter_um"] = df["equivalent_diameter_px"] * PIXEL_SIZE_UM
    df["keep"] = df["equivalent_diameter_px"] >= MIN_DIAMETER_PX

    keep_labels = df.loc[df["keep"], "label"].to_numpy(dtype=np.int32)

    inst_filt = np.zeros_like(inst, dtype=np.int32)
    if keep_labels.size > 0:
        keep_mask = np.isin(inst, keep_labels)
        inst_filt = np.where(keep_mask, inst, 0).astype(np.int32)
        inst_filt, _, _ = relabel_sequential(inst_filt)

    return inst_filt, df


# ----------------------------
# Per-mosaic processing
# ----------------------------
def reconstruct_segdata_and_filtered_instance(seg_dir: Path, out_dir: Path) -> None:
    mask_tiles = list_mask_tiles(seg_dir)
    if not mask_tiles:
        raise RuntimeError(f"No SegData tiles found in {seg_dir}")

    tile_ids = sorted(mask_tiles.keys())
    nrows, ncols = infer_grid(len(tile_ids))

    first_mask = read_image(mask_tiles[tile_ids[0]])
    h, w = first_mask.shape[:2]

    seg_mosaic = np.zeros((nrows * h, ncols * w), dtype=np.uint16)
    inst_mosaic = np.zeros((nrows * h, ncols * w), dtype=np.uint32)

    label_offset = 0
    instance_rows = []

    for tid in tile_ids:
        mask = read_image(mask_tiles[tid])
        raw_labels = build_labeled_mask(mask)
        filt_labels, df = filter_labels_by_diameter(raw_labels)

        row, col = snake_position(tid, ncols)
        r0, r1 = row * h, (row + 1) * h
        c0, c1 = col * w, (col + 1) * w

        seg_mosaic[r0:r1, c0:c1] = (mask > 0).astype(np.uint16)

        if filt_labels.max() > 0:
            filt_labels = filt_labels.astype(np.uint32)
            filt_labels[filt_labels > 0] += label_offset
            label_offset = int(filt_labels.max())

        inst_mosaic[r0:r1, c0:c1] = filt_labels

        if len(df) > 0:
            df = df.copy()
            df.insert(0, "tile_id", tid)
            instance_rows.append(df)

    save_tiff(out_dir / "segdata_mosaic.tif", seg_mosaic)
    save_png(out_dir / "segdata_mosaic.png", seg_mosaic)

    save_tiff(out_dir / "instance_mask_filtered.tif", inst_mosaic)
    save_png(out_dir / "instance_mask_filtered.png", inst_mosaic > 0)

    if instance_rows:
        pd.concat(instance_rows, ignore_index=True).to_csv(
            out_dir / "instance_mask_filtered_props.csv", index=False
        )


def process_mosaic_flim(mosaic_dir: Path, ref_mean: np.ndarray, ref_real: np.ndarray, ref_imag: np.ndarray) -> None:
    flim_dir = mosaic_dir / "flim"
    seg_dir = mosaic_dir / "SegData"
    out_dir = mosaic_dir / "_new"
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = mode_from_folder_name(mosaic_dir.name)

    flim_tiles = list_flim_tiles(flim_dir)
    if not flim_tiles:
        raise RuntimeError(f"No FLIM tiles found in {flim_dir}")

    tile_ids = sorted(flim_tiles.keys())
    nrows, ncols = infer_grid(len(tile_ids))

    # process first tile to define shape
    first_signal = load_flim_signal(flim_tiles[tile_ids[0]], mode=mode)
    mean0, real0, imag0 = phasor_from_signal(first_signal, harmonic=[1, 2], axis=0)
    real0_cal, imag0_cal = phasor_calibrate(
        real0, imag0,
        ref_mean, ref_real, ref_imag,
        frequency=FREQUENCY_MHZ,
        harmonic=[1, 2],
        lifetime=COUMARIN_LIFETIME_NS,
    )

    h, w = mean0.shape

    avg_mosaic = np.zeros((nrows * h, ncols * w), dtype=np.float32)
    g1_mosaic = np.zeros_like(avg_mosaic)
    s1_mosaic = np.zeros_like(avg_mosaic)
    g2_mosaic = np.zeros_like(avg_mosaic)
    s2_mosaic = np.zeros_like(avg_mosaic)

    def place(tile_id: int, mean: np.ndarray, real: np.ndarray, imag: np.ndarray) -> None:
        row, col = snake_position(tile_id, ncols)
        r0, r1 = row * h, (row + 1) * h
        c0, c1 = col * w, (col + 1) * w

        avg_mosaic[r0:r1, c0:c1] = mean.astype(np.float32)
        g1_mosaic[r0:r1, c0:c1] = real[0].astype(np.float32)
        s1_mosaic[r0:r1, c0:c1] = imag[0].astype(np.float32)
        g2_mosaic[r0:r1, c0:c1] = real[1].astype(np.float32)
        s2_mosaic[r0:r1, c0:c1] = imag[1].astype(np.float32)

    place(tile_ids[0], mean0, real0_cal, imag0_cal)

    for tid in tile_ids[1:]:
        signal = load_flim_signal(flim_tiles[tid], mode=mode)
        mean, real, imag = phasor_from_signal(signal, harmonic=[1, 2], axis=0)
        real_cal, imag_cal = phasor_calibrate(
            real, imag,
            ref_mean, ref_real, ref_imag,
            frequency=FREQUENCY_MHZ,
            harmonic=[1, 2],
            lifetime=COUMARIN_LIFETIME_NS,
        )
        place(tid, mean, real_cal, imag_cal)

    # save only stack + avg png
    phasor_stack = np.stack([avg_mosaic, g1_mosaic, s1_mosaic, g2_mosaic, s2_mosaic], axis=0)
    save_tiff(out_dir / "phasor.tif", phasor_stack)
    save_png(out_dir / "avg.png", avg_mosaic)

    # reconstruct SegData and filtered instance mask
    reconstruct_segdata_and_filtered_instance(seg_dir, out_dir)

    print(f"[OK] Processed mosaic: {mosaic_dir.name}")


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    DEFAULT_VISIT_DIR = Path(
        "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449/visit04"
    ).expanduser()

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--visit-dir",
        type=str,
        default=str(DEFAULT_VISIT_DIR),
        help="Visit folder containing one coumarin_* folder and multiple mosaic folders",
    )
    args = ap.parse_args()

    visit_dir = Path(args.visit_dir).expanduser().resolve()
    if not visit_dir.exists():
        raise FileNotFoundError(f"Visit folder not found: {visit_dir}")

    coumarin_dir = find_coumarin_dir(visit_dir)
    mosaic_dirs = find_mosaic_dirs(visit_dir)

    if not mosaic_dirs:
        raise RuntimeError(f"No mosaic folders with flim/ and SegData/ found in {visit_dir}")

    print(f"[INFO] Visit: {visit_dir.name}")
    print(f"[INFO] Coumarin folder: {coumarin_dir.name}")
    print(f"[INFO] Mosaics found: {len(mosaic_dirs)}")

    ref_mean, ref_real, ref_imag = build_reference_phasor(coumarin_dir)

    for mosaic_dir in mosaic_dirs:
        process_mosaic_flim(mosaic_dir, ref_mean, ref_real, ref_imag)

    print("[DONE] Visit processing finished.")


if __name__ == "__main__":
    main()