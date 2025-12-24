"""
calculate_phasor.py
===================

Goal
----
Compute per-pixel phasor maps (H1 + H2) from raw FLIM mosaic tiles and save:
1) uncalibrated phasor maps
2) coumarin-calibrated phasor maps

This script is designed for the UCI / BLI melanoma FLIM workflow where each visit folder
contains:
- exactly one coumarin reference folder (name starts with: "coumarin_")
- one or more mosaic folders (name starts with: "Mosaic")
  each mosaic folder contains 16 FLIM tile files:
    Im_00001.tif ... Im_00016.tif

The 16 tiles form a 4x4 mosaic WITHOUT overlap. The scan order is serpentine:
Row1:  1,  2,  3,  4
Row2:  8,  7,  6,  5
Row3:  9, 10, 11, 12
Row4: 16, 15, 14, 13

What is computed
----------------
For each mosaic:
- mean intensity image (mean of selected decay bins)
- phasor real/imag for harmonic 1 and harmonic 2 (g,s)
- phasor polar coordinates: modulation and phase for H1 and H2

Then calibration:
- Uses coumarin reference measurement (known lifetime = 2.5 ns) at laser frequency 80 MHz
- Computes coumarin reference phasor as an intensity-weighted center-of-mass (COM)
- Applies `phasorpy.lifetime.phasor_calibrate` to correct IRF/delay effects

Decay bin selection (VERY IMPORTANT)
------------------------------------
This script can select only a subset of time bins from the raw FLIM decay.
Current default is:

    T_START = 0
    T_END   = 16

Meaning: use ONLY bins [0..15] (16 bins total).
This is intentional for your current dataset/strategy ("use only the FIRST decay").

Output files
------------
For each mosaic, the script writes two multi-channel TIFFs (float32) with axes "CYX"
(C=channels, Y,X spatial):

    phasor_uncalibrated_CYX.tif
    phasor_calibrated_CYX.tif

Saved under:
    OUT_ROOT / <patient_dir_name> / <visit_XX> / <Mosaic...> /

Channel layout (C axis)
-----------------------
C=0:  mean intensity (selected decay bins)
C=1:  g1  (real H1)
C=2:  s1  (imag H1)
C=3:  mod1
C=4:  phase1
C=5:  g2  (real H2)
C=6:  s2  (imag H2)
C=7:  mod2
C=8:  phase2

How to run
----------
Edit PATIENT_DIR and OUT_ROOT below, then:
    python calculate_phasor.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import tifffile as tiff

from phasorpy.phasor import phasor_from_signal, phasor_to_polar
from phasorpy.lifetime import phasor_calibrate


# ============================================================
# CONFIG (EDIT THESE)
# ============================================================
PATIENT_DIR = Path("/Users/schutyb/Documents/balu_lab/data_patient_449")
OUT_ROOT = Path("/Users/schutyb/Documents/balu_lab/data_patient_449/phasor_out")

FREQUENCY_MHZ = 80.0
COUMARIN_LIFETIME_NS = 2.5

# Decay bins to use for the phasor calculation
# Default: use ONLY bins 0..15 (16 bins total)
T_START = 0
T_END = 16  # slice is [T_START:T_END] (Python-style, T_END exclusive)

# 4x4 serpentine tile order (how the 16 tiles are arranged in the mosaic)
TILE_ORDER_4x4 = np.array(
    [
        [1, 2, 3, 4],
        [8, 7, 6, 5],
        [9, 10, 11, 12],
        [16, 15, 14, 13],
    ],
    dtype=int,
)


# ============================================================
# Robust TIFF loading -> returns array, then converted to (T,Y,X)
# ============================================================
def load_tiff_robust(path: Path) -> np.ndarray:
    """
    Robust TIFF loader that avoids tifffile trying to reshape data into unexpected dims.

    Strategy:
    - First try `tifffile.imread` normally.
    - If tifffile throws due to unexpected series/reshape assumptions, fall back to:
        open TiffFile -> read each page -> stack into (pages, Y, X).

    Returns
    -------
    arr : np.ndarray
        Raw loaded data (often 3D).
    """
    try:
        return np.asarray(tiff.imread(str(path)))
    except Exception:
        with tiff.TiffFile(str(path)) as tf:
            pages = [p.asarray() for p in tf.pages]
        return np.stack(pages, axis=0)


def to_tyx(arr: np.ndarray) -> np.ndarray:
    """
    Convert common raw tile formats to a standard (T, Y, X) float32 array.

    Accepts one of:
      - (T, Y, X)
      - (Y, X, T)
      - (Y, T, X)

    Heuristics:
    - The "time" axis is typically the smallest (<=256) and spatial axes are >=16.
    """
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D tile. Got shape {arr.shape}")

    a, b, c = arr.shape

    # Typical: (T,Y,X)
    if a <= 256 and b >= 16 and c >= 16:
        return arr.astype(np.float32)

    # (Y,X,T) -> (T,Y,X)
    if c <= 256 and a >= 16 and b >= 16:
        return np.moveaxis(arr, -1, 0).astype(np.float32)

    # (Y,T,X) -> (T,Y,X)
    if b <= 256 and a >= 16 and c >= 16:
        return np.moveaxis(arr, 1, 0).astype(np.float32)

    # Fallback: leave as-is (still float32)
    return arr.astype(np.float32)


def load_tile_tyx(path: Path) -> np.ndarray:
    """Load a single FLIM tile and return as (T,Y,X) float32."""
    return to_tyx(load_tiff_robust(path))


# ============================================================
# Folder helpers
# ============================================================
def find_single_subdir(parent: Path, prefix: str) -> Path:
    """
    Return a subdirectory in `parent` starting with `prefix`.
    If multiple matches exist, picks the newest by modified time.
    """
    matches = sorted([p for p in parent.iterdir() if p.is_dir() and p.name.startswith(prefix)])
    if len(matches) == 0:
        raise FileNotFoundError(f"No subfolder starting with '{prefix}' in: {parent}")
    if len(matches) > 1:
        matches = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def list_mosaic_dirs(visit_dir: Path) -> list[Path]:
    """List Mosaic* folders inside a visit folder."""
    return sorted([p for p in visit_dir.iterdir() if p.is_dir() and p.name.startswith("Mosaic")])


# ============================================================
# Mosaic assembly -> (T,Y,X)
# ============================================================
def assemble_mosaic_4x4_tyx(tiles: dict[int, np.ndarray]) -> np.ndarray:
    """
    Assemble 16 tiles into a single 4x4 mosaic.

    tiles: dict {1..16 -> (T,ty,tx)}
    returns: (T, 4*ty, 4*tx)
    """
    missing = [i for i in range(1, 17) if i not in tiles]
    if missing:
        raise FileNotFoundError(f"Missing tiles: {missing}")

    shapes = {tiles[i].shape for i in range(1, 17)}
    if len(shapes) != 1:
        raise ValueError(f"Tile shapes differ: {shapes}")

    T, ty, tx = next(iter(shapes))
    mosaic = np.zeros((T, ty * 4, tx * 4), dtype=np.float32)

    for r in range(4):
        for c in range(4):
            idx = int(TILE_ORDER_4x4[r, c])
            y0, y1 = r * ty, (r + 1) * ty
            x0, x1 = c * tx, (c + 1) * tx
            mosaic[:, y0:y1, x0:x1] = tiles[idx]

    return mosaic


def load_mosaic_stack_tyx(mosaic_dir: Path) -> np.ndarray:
    """Load Im_00001..Im_00016 from a Mosaic folder and assemble into (T,Y,X)."""
    tiles: dict[int, np.ndarray] = {}
    for i in range(1, 17):
        fpath = mosaic_dir / f"Im_{i:05d}.tif"
        if not fpath.exists():
            raise FileNotFoundError(f"Missing tile: {fpath}")
        tiles[i] = load_tile_tyx(fpath)
    return assemble_mosaic_4x4_tyx(tiles)


# ============================================================
# Phasor compute + calibration
# ============================================================
def phasor_maps_from_raw_tyx(
    raw_tyx: np.ndarray,
    harmonics=(1, 2),
    t_start: int = T_START,
    t_end: int | None = T_END,
):
    """
    Compute phasor maps from a (T,Y,X) raw FLIM stack using selected decay bins.

    Returns dict:
      mean  : (Y,X)
      real  : (H,Y,X)
      imag  : (H,Y,X)
      mod   : (H,Y,X)
      phase : (H,Y,X)
    """
    T = raw_tyx.shape[0]
    t_start_eff = max(int(t_start), 0)
    t_end_eff = T if t_end is None else min(int(t_end), T)

    if t_start_eff >= t_end_eff:
        raise ValueError(f"Invalid decay slice: start={t_start_eff}, end={t_end_eff}, T={T}")

    decay = raw_tyx[t_start_eff:t_end_eff].astype(np.float32)  # (Tsel,Y,X)

    mean, real, imag = phasor_from_signal(decay, axis=0, harmonic=list(harmonics))
    phase, mod = phasor_to_polar(real, imag)  # (phase, modulation)

    return {
        "mean": mean.astype(np.float32),
        "real": real.astype(np.float32),
        "imag": imag.astype(np.float32),
        "mod": mod.astype(np.float32),
        "phase": phase.astype(np.float32),
    }


def reference_phasor_from_coumarin(
    coumarin_dir: Path,
    harmonics=(1, 2),
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Compute coumarin reference phasor (intensity-weighted COM) from coumarin tiles.

    Returns: (ref_mean, ref_real(H,), ref_imag(H,))
    """
    tifs = sorted(list(coumarin_dir.glob("*.tif")) + list(coumarin_dir.glob("*.tiff")))
    if not tifs:
        raise FileNotFoundError(f"No .tif/.tiff in coumarin folder: {coumarin_dir}")

    w_sum = 0.0
    real_acc = None
    imag_acc = None
    mean_acc = 0.0

    for fp in tifs:
        sig_tyx = load_tile_tyx(fp)
        maps = phasor_maps_from_raw_tyx(sig_tyx, harmonics=harmonics)

        m = maps["mean"]   # (Y,X)
        r = maps["real"]   # (H,Y,X)
        im = maps["imag"]  # (H,Y,X)

        w = float(np.nansum(m))
        if w <= 0:
            continue

        mr = np.nansum(r * m[None, :, :], axis=(1, 2))
        mi = np.nansum(im * m[None, :, :], axis=(1, 2))
        r_com = mr / w
        i_com = mi / w
        m_mean = float(np.nanmean(m))

        if real_acc is None:
            real_acc = r_com * w
            imag_acc = i_com * w
        else:
            real_acc += r_com * w
            imag_acc += i_com * w

        mean_acc += m_mean * w
        w_sum += w

    if w_sum <= 0 or real_acc is None:
        raise RuntimeError(f"Coumarin reference could not be computed in: {coumarin_dir}")

    ref_real = (real_acc / w_sum).astype(np.float32)
    ref_imag = (imag_acc / w_sum).astype(np.float32)
    ref_mean = float(mean_acc / w_sum)
    return ref_mean, ref_real, ref_imag


def calibrate_phasor_maps(maps, ref_mean, ref_real_h, ref_imag_h, harmonics=(1, 2)):
    """Apply phasor calibration using coumarin reference."""
    real = maps["real"]  # (H,Y,X)
    imag = maps["imag"]
    H, Y, X = real.shape

    ref_real = np.broadcast_to(ref_real_h[:, None, None], (H, Y, X)).astype(np.float32)
    ref_imag = np.broadcast_to(ref_imag_h[:, None, None], (H, Y, X)).astype(np.float32)
    ref_mean_arr = np.full((Y, X), ref_mean, dtype=np.float32)

    real_c, imag_c = phasor_calibrate(
        real,
        imag,
        ref_mean_arr,
        ref_real,
        ref_imag,
        frequency=FREQUENCY_MHZ,
        lifetime=COUMARIN_LIFETIME_NS,
        harmonic=list(harmonics),
        nan_safe=True,
    )

    phase_c, mod_c = phasor_to_polar(real_c, imag_c)

    out = dict(maps)
    out["real"] = np.asarray(real_c, dtype=np.float32)
    out["imag"] = np.asarray(imag_c, dtype=np.float32)
    out["mod"] = np.asarray(mod_c, dtype=np.float32)
    out["phase"] = np.asarray(phase_c, dtype=np.float32)
    return out


# ============================================================
# Save TIFF (CYX)
# ============================================================
def pack_channels_cyx(maps: dict) -> np.ndarray:
    """
    Pack phasor maps into CYX channel layout (float32):

      0 mean
      1 g1
      2 s1
      3 mod1
      4 phase1
      5 g2
      6 s2
      7 mod2
      8 phase2
    """
    mean = maps["mean"]
    real = maps["real"]
    imag = maps["imag"]
    mod = maps["mod"]
    phase = maps["phase"]

    if real.shape[0] < 2 or imag.shape[0] < 2:
        raise ValueError("Expected harmonics=(1,2) so real/imag have H>=2")

    return np.stack(
        [
            mean,
            real[0], imag[0], mod[0], phase[0],
            real[1], imag[1], mod[1], phase[1],
        ],
        axis=0,
    ).astype(np.float32)


def save_tiff_cyx(path: Path, cyx: np.ndarray):
    """Write a float32 TIFF with axes metadata 'CYX'."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(str(path), cyx, photometric="minisblack", metadata={"axes": "CYX"})


# ============================================================
# Main driver
# ============================================================
def process_patient(patient_dir: Path, out_root: Path):
    """Traverse patient_dir/visit_*/Mosaic* and compute phasor TIFFs."""
    visits = sorted([p for p in patient_dir.iterdir() if p.is_dir() and p.name.startswith("visit_")])
    if not visits:
        raise FileNotFoundError(f"No visit_* folders in: {patient_dir}")

    for visit_dir in visits:
        print(f"\n[VISIT] {visit_dir.name}")

        coumarin_dir = find_single_subdir(visit_dir, "coumarin_")
        print(f"  Coumarin folder: {coumarin_dir.name}")

        ref_mean, ref_real_h, ref_imag_h = reference_phasor_from_coumarin(coumarin_dir, harmonics=(1, 2))
        print(f"  Coumarin ref mean: {ref_mean:.4f}")

        mosaic_dirs = list_mosaic_dirs(visit_dir)
        if not mosaic_dirs:
            print("  [WARN] No Mosaic* folders found.")
            continue

        for mosaic_dir in mosaic_dirs:
            print(f"  [MOSAIC] {mosaic_dir.name}")

            raw_tyx = load_mosaic_stack_tyx(mosaic_dir)
            print(f"    Raw mosaic shape (T,Y,X): {raw_tyx.shape}")

            bins_used = (T_END - T_START) if T_END is not None else (raw_tyx.shape[0] - T_START)
            print(f"    Using decay bins slice: [{T_START}:{T_END}] -> {bins_used} bins")

            maps_uncal = phasor_maps_from_raw_tyx(raw_tyx, harmonics=(1, 2))
            maps_cal = calibrate_phasor_maps(maps_uncal, ref_mean, ref_real_h, ref_imag_h, harmonics=(1, 2))

            out_dir = out_root / patient_dir.name / visit_dir.name / mosaic_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)

            save_tiff_cyx(out_dir / "phasor_uncalibrated_CYX.tif", pack_channels_cyx(maps_uncal))
            save_tiff_cyx(out_dir / "phasor_calibrated_CYX.tif", pack_channels_cyx(maps_cal))

            print("    [SAVED] uncal + calibrated:", out_dir)


def main():
    # Minimal checks
    if not PATIENT_DIR.exists():
        raise FileNotFoundError(f"PATIENT_DIR does not exist: {PATIENT_DIR}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] PATIENT_DIR: {PATIENT_DIR}")
    print(f"[INFO] OUT_ROOT:    {OUT_ROOT}")
    process_patient(PATIENT_DIR, OUT_ROOT)


if __name__ == "__main__":
    main()