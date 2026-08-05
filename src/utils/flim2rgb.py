#!/usr/bin/env python3
"""Create pseudo-RGB mosaics from simultaneous Sp and sequential A1/A0 FLIM.

The script receives a data root containing several patient folders and handles
two acquisition formats:

``Sp`` (simultaneous split acquisition)
    The tile stack contains both detectors. Green uses the first 16 temporal
    bins and blue uses the remaining 15 or 16 bins. The pseudo-RGB mapping is
    R=first 4 green bins, G=remaining green bins, B=all blue bins.

``A1/A0`` (sequential single-detector acquisitions)
    A1 is green and A0 is blue. Each detector is normalized to 32 temporal
    bins by copying bin 31 into bin 32. A1 uses R=first 4 bins and G=remaining
    28 bins; A0 uses B=all 32 bins. The blue mosaic is registered to the green
    mosaic using a global translation, then all channels are mean-binned for
    visualization (default 5x5).

``A1 only``
    If no matching A0 acquisition exists, an explicitly labeled green-only
    pseudo-RGB is generated with B=0 and the same visualization binning.

Sequential pseudo-RGB images are visualization/QC products, not quantitative
colocalization measurements. Registration metadata and before/after QC images
are saved with every sequential result.

Example
-------
.venv/bin/python src/utils/flim2rgb.py \
    --data-root /Users/schutyb/Documents/balu_lab/dod/data_curated \
    --patients p439 p427
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import traceback
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel
from scipy.ndimage import shift as ndi_shift
from skimage.registration import phase_cross_correlation

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_DATA_ROOT = Path(
    "/Users/schutyb/Documents/balu_lab/dod/data_curated"
)

FLIM_SUBDIR = "flim"
RGB_OUTPUT_SUBDIR = "RGB"

SP_GREEN_BINS = 16
SINGLE_DETECTOR_BINS = 32
ACCEPTED_STACK_BINS = (31, 32)

# Pseudo-RGB channel gains applied before percentile normalization.
SCALE_R = 4.0 / 3.0
SCALE_G = 3.0 / 3.0
SCALE_B = 2.0 / 3.0

PERCENTILE_LOW = 1.0
PERCENTILE_HIGH = 99.0

SEQUENTIAL_BIN_SIZE = 5
REGISTER_SEQUENTIAL = True
REGISTRATION_DOWNSAMPLE = 4
REGISTRATION_UPSAMPLE_FACTOR = 10
REGISTRATION_GAUSSIAN_SIGMA = 1.5
MAX_ABS_REGISTRATION_SHIFT_PIXELS = 60.0
REGISTRATION_QC_MAX_DIMENSION = 1200

PNG_DPI = 600
OVERWRITE = True
CONTINUE_ON_ERROR = True


def natural_key(value: str | Path) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", str(value))
    return tuple(int(part) if part.isdigit() else part.casefold() for part in parts)


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return cleaned.strip("._") or "mosaic"


def find_patient_directories(
    data_root: Path,
    requested_patients: list[str] | None,
) -> list[Path]:
    if requested_patients:
        patients = [data_root / name for name in requested_patients]
        missing = [path for path in patients if not path.is_dir()]
        if missing:
            text = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(f"Patient directories not found:\n{text}")
        return patients

    return sorted(
        (
            path
            for path in data_root.iterdir()
            if path.is_dir() and re.fullmatch(r"p\d+", path.name, re.IGNORECASE)
        ),
        key=natural_key,
    )


def acquisition_type(mosaic_name: str) -> str | None:
    if re.search(r"sp$", mosaic_name, re.IGNORECASE):
        return "sp"
    if re.search(r"a1$", mosaic_name, re.IGNORECASE):
        return "green_a1"
    if re.search(r"a0$", mosaic_name, re.IGNORECASE):
        return "blue_a0"
    return None


def sequential_pair_key(mosaic_dir: Path) -> tuple[str, str]:
    """Pair A1/A0 while ignoring MosaicXX and the detector suffix."""
    normalized = re.sub(
        r"^mosaic\d+",
        "mosaic",
        mosaic_dir.name,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"a[01]$", "a", normalized, flags=re.IGNORECASE)
    return str(mosaic_dir.parent.resolve()).casefold(), normalized.casefold()


def classify_visit_mosaics(
    visit_dir: Path,
) -> tuple[list[Path], list[tuple[Path, Path]], list[Path], list[str]]:
    # Some patients store acquisitions directly below visitXX, while others
    # use an intermediate folder such as Mosaic_visit01_z110. Search the whole
    # visit tree and retain only directories whose own names identify an
    # acquisition (Sp, A1, or A0).
    mosaic_dirs = sorted(
        (
            path
            for path in visit_dir.rglob("*")
            if path.is_dir()
            and path.name.casefold().startswith("mosaic")
            and acquisition_type(path.name) is not None
        ),
        key=natural_key,
    )
    sp_dirs: list[Path] = []
    green_groups: dict[tuple[str, str], list[Path]] = {}
    blue_groups: dict[tuple[str, str], list[Path]] = {}
    warnings: list[str] = []

    for mosaic_dir in mosaic_dirs:
        kind = acquisition_type(mosaic_dir.name)
        if kind == "sp":
            sp_dirs.append(mosaic_dir)
        elif kind == "green_a1":
            green_groups.setdefault(sequential_pair_key(mosaic_dir), []).append(
                mosaic_dir
            )
        elif kind == "blue_a0":
            blue_groups.setdefault(sequential_pair_key(mosaic_dir), []).append(
                mosaic_dir
            )
        else:
            raise RuntimeError(
                f"Internal acquisition classification failure: {mosaic_dir}"
            )

    pairs: list[tuple[Path, Path]] = []
    green_only: list[Path] = []
    all_keys = sorted(set(green_groups) | set(blue_groups), key=natural_key)
    for key in all_keys:
        greens = sorted(green_groups.get(key, []), key=natural_key)
        blues = sorted(blue_groups.get(key, []), key=natural_key)
        pair_count = min(len(greens), len(blues))
        pairs.extend(zip(greens[:pair_count], blues[:pair_count], strict=False))
        green_only.extend(greens[pair_count:])
        for path in blues[pair_count:]:
            warnings.append(f"A0 acquisition without matching A1: {path}")

    return sp_dirs, pairs, sorted(green_only, key=natural_key), warnings


def parse_mosaic_shape(folder_name: str) -> tuple[int, int]:
    match = re.search(r"(\d+)x(\d+)", folder_name)
    if match is None:
        raise ValueError(f"Mosaic grid not found in folder name: {folder_name}")
    return int(match.group(1)), int(match.group(2))


def collect_tiles(flim_dir: Path) -> dict[int, Path]:
    tiles: dict[int, Path] = {}
    for path in list(flim_dir.glob("Im_*.tif")) + list(flim_dir.glob("Im_*.tiff")):
        match = re.fullmatch(r"Im_(\d+)\.tiff?", path.name, re.IGNORECASE)
        if match:
            tile_number = int(match.group(1))
            if tile_number in tiles:
                raise ValueError(f"Duplicate tile number {tile_number}: {flim_dir}")
            tiles[tile_number] = path
    return dict(sorted(tiles.items()))


def locate_flim_tiles(mosaic_dir: Path) -> tuple[Path, dict[int, Path]]:
    """Locate tiles in ``flim``, ``flim ``, or the mosaic directory itself."""
    child_candidates = sorted(
        (
            path
            for path in mosaic_dir.iterdir()
            if path.is_dir() and path.name.strip().casefold() == FLIM_SUBDIR
        ),
        key=lambda path: (path.name != FLIM_SUBDIR, natural_key(path)),
    )
    candidates = [*child_candidates, mosaic_dir]
    for candidate in candidates:
        tiles = collect_tiles(candidate)
        if tiles:
            return candidate, tiles
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No Im_*.tif tiles found; searched: {searched}")


def read_flim_stack(path: Path) -> np.ndarray:
    """Read TIFF pages explicitly, ignoring inconsistent shaped metadata."""
    with tifffile.TiffFile(path) as tif:
        if len(tif.pages) not in ACCEPTED_STACK_BINS:
            raise ValueError(
                f"Expected {ACCEPTED_STACK_BINS} TIFF pages in {path}, "
                f"found {len(tif.pages)}"
            )
        return tif.asarray(key=slice(None))


def ensure_yxt(stack: np.ndarray, accepted_bins: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(stack)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D FLIM stack, found shape {array.shape}")

    candidate_axes = [
        axis for axis, size in enumerate(array.shape) if size in accepted_bins
    ]
    if len(candidate_axes) != 1:
        raise ValueError(
            f"Could not identify a unique bin axis in shape {array.shape}; "
            f"accepted bin counts are {accepted_bins}"
        )
    return np.moveaxis(array, candidate_axes[0], -1).astype(np.float32, copy=False)


def normalize_single_detector_32(stack: np.ndarray) -> np.ndarray:
    """Return Y,X,32 and always copy one-based bin 31 into bin 32."""
    array = ensure_yxt(stack, ACCEPTED_STACK_BINS)
    if array.shape[-1] == 31:
        array = np.concatenate((array, array[..., 30:31]), axis=-1)
    else:
        array = np.array(array, copy=True)
        array[..., 31] = array[..., 30]
    return array


def split_sp_stack(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return corrected green and blue arrays with 16 bins each."""
    array = ensure_yxt(stack, ACCEPTED_STACK_BINS)
    green = np.array(array[..., :SP_GREEN_BINS], copy=True)
    blue = np.array(array[..., SP_GREEN_BINS:], copy=True)

    if green.shape[-1] != SP_GREEN_BINS:
        raise ValueError(f"Sp green channel has {green.shape[-1]} bins, expected 16")
    # Match the established split correction: green bin 16 <- bin 15.
    green[..., 15] = green[..., 14]

    if blue.shape[-1] == 15:
        blue = np.concatenate((blue, blue[..., -1:]), axis=-1)
    elif blue.shape[-1] == 16:
        blue[..., 15] = blue[..., 14]
    else:
        raise ValueError(f"Sp blue channel has {blue.shape[-1]} bins, expected 15/16")
    return green, blue


def sp_components(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    green, blue = split_sp_stack(stack)
    red = green[..., :4].sum(axis=-1, dtype=np.float32) * SCALE_R
    green_image = green[..., 4:].sum(axis=-1, dtype=np.float32) * SCALE_G
    blue_image = blue.sum(axis=-1, dtype=np.float32) * SCALE_B
    return red, green_image, blue_image


def a1_components(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    green = normalize_single_detector_32(stack)
    red = green[..., :4].sum(axis=-1, dtype=np.float32) * SCALE_R
    green_image = green[..., 4:].sum(axis=-1, dtype=np.float32) * SCALE_G
    return red, green_image


def a0_component(stack: np.ndarray) -> np.ndarray:
    blue = normalize_single_detector_32(stack)
    return blue.sum(axis=-1, dtype=np.float32) * SCALE_B


def snake_position(tile_number: int, rows: int, cols: int) -> tuple[int, int]:
    index = tile_number - 1
    row = index // cols
    scan_col = index % cols
    if row >= rows:
        raise ValueError(f"Tile {tile_number} exceeds mosaic grid {rows}x{cols}")
    col = scan_col if row % 2 == 0 else cols - 1 - scan_col
    return row, col


def reconstruct_float_mosaic(
    tile_images: dict[int, np.ndarray],
    rows: int,
    cols: int,
) -> np.ndarray:
    expected_numbers = set(range(1, rows * cols + 1))
    if set(tile_images) != expected_numbers:
        missing = sorted(expected_numbers.difference(tile_images))
        extra = sorted(set(tile_images).difference(expected_numbers))
        raise ValueError(f"Tile mismatch; missing={missing}, extra={extra}")

    first = tile_images[1]
    height, width = first.shape
    mosaic = np.zeros((rows * height, cols * width), dtype=np.float32)
    for tile_number, tile in tile_images.items():
        if tile.shape != (height, width):
            raise ValueError(
                f"Tile shape mismatch: tile {tile_number}={tile.shape}, "
                f"expected={(height, width)}"
            )
        row, col = snake_position(tile_number, rows, cols)
        y0, x0 = row * height, col * width
        mosaic[y0 : y0 + height, x0 : x0 + width] = tile
    return mosaic


def channel_percentiles(channel: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(channel)
    if not np.any(valid):
        return 0.0, 0.0
    low, high = np.percentile(
        channel[valid],
        (PERCENTILE_LOW, PERCENTILE_HIGH),
    )
    return float(low), float(high)


def normalize_channel(
    channel: np.ndarray,
    low_value: float,
    high_value: float,
) -> np.ndarray:
    if high_value <= low_value:
        return np.zeros_like(channel, dtype=np.float32)
    normalized = np.clip(channel, low_value, high_value)
    normalized = (normalized - low_value) / (high_value - low_value)
    return normalized.astype(np.float32, copy=False)


def build_rgb(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
) -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    percentiles = {
        "R": channel_percentiles(red),
        "G": channel_percentiles(green),
        "B": channel_percentiles(blue),
    }
    rgb = np.stack(
        (
            normalize_channel(red, *percentiles["R"]),
            normalize_channel(green, *percentiles["G"]),
            normalize_channel(blue, *percentiles["B"]),
        ),
        axis=-1,
    )
    return (np.clip(rgb, 0, 1) * 255).round().astype(np.uint8), percentiles


def save_png(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(
        path,
        dpi=(PNG_DPI, PNG_DPI),
        optimize=True,
    )


def mean_bin_image(image: np.ndarray, bin_size: int) -> np.ndarray:
    if bin_size <= 1:
        return np.asarray(image, dtype=np.float32)
    height = image.shape[0] - image.shape[0] % bin_size
    width = image.shape[1] - image.shape[1] % bin_size
    if height == 0 or width == 0:
        raise ValueError(f"Bin size {bin_size} is too large for shape {image.shape}")
    cropped = np.asarray(image[:height, :width], dtype=np.float32)
    return cropped.reshape(
        height // bin_size,
        bin_size,
        width // bin_size,
        bin_size,
    ).mean(axis=(1, 3), dtype=np.float32)


def registration_feature(image: np.ndarray, downsample: int) -> np.ndarray:
    normalized = np.log1p(np.maximum(np.asarray(image, dtype=np.float32), 0))
    low, high = np.percentile(normalized[np.isfinite(normalized)], (1, 99))
    normalized = normalize_channel(normalized, float(low), float(high))
    normalized = gaussian_filter(normalized, sigma=REGISTRATION_GAUSSIAN_SIGMA)
    gradient_y = sobel(normalized, axis=0)
    gradient_x = sobel(normalized, axis=1)
    gradient = np.hypot(gradient_x, gradient_y).astype(np.float32)
    if downsample > 1:
        gradient = mean_bin_image(gradient, downsample)
    gradient -= float(np.mean(gradient))
    return gradient


def estimate_translation(
    reference_green: np.ndarray,
    moving_blue: np.ndarray,
) -> dict[str, Any]:
    if reference_green.shape != moving_blue.shape:
        raise ValueError(
            f"Registration shape mismatch: green={reference_green.shape}, "
            f"blue={moving_blue.shape}"
        )
    reference_feature = registration_feature(
        reference_green,
        REGISTRATION_DOWNSAMPLE,
    )
    moving_feature = registration_feature(
        moving_blue,
        REGISTRATION_DOWNSAMPLE,
    )
    shift_small, error, phase_difference = phase_cross_correlation(
        reference_feature,
        moving_feature,
        upsample_factor=REGISTRATION_UPSAMPLE_FACTOR,
        normalization=None,
    )
    shift_y = float(shift_small[0] * REGISTRATION_DOWNSAMPLE)
    shift_x = float(shift_small[1] * REGISTRATION_DOWNSAMPLE)
    if not np.isfinite(shift_y) or not np.isfinite(shift_x):
        raise ValueError(
            f"Registration returned non-finite shift: {(shift_y, shift_x)}"
        )
    if max(abs(shift_y), abs(shift_x)) > MAX_ABS_REGISTRATION_SHIFT_PIXELS:
        raise ValueError(
            "Estimated sequential shift exceeds safety limit: "
            f"dy={shift_y:.3f}, dx={shift_x:.3f}, "
            f"limit={MAX_ABS_REGISTRATION_SHIFT_PIXELS:.1f} pixels"
        )
    return {
        "shift_y_pixels": shift_y,
        "shift_x_pixels": shift_x,
        "registration_error": float(error),
        "phase_difference": float(phase_difference),
        "registration_downsample": REGISTRATION_DOWNSAMPLE,
        "registration_upsample_factor": REGISTRATION_UPSAMPLE_FACTOR,
    }


def apply_translation(image: np.ndarray, shift_y: float, shift_x: float) -> np.ndarray:
    return ndi_shift(
        np.asarray(image, dtype=np.float32),
        shift=(shift_y, shift_x),
        order=1,
        mode="nearest",
        prefilter=False,
    ).astype(np.float32, copy=False)


def qc_display_image(image: np.ndarray) -> np.ndarray:
    low, high = channel_percentiles(image)
    return normalize_channel(image, low, high)


def save_registration_qc(
    green_total: np.ndarray,
    blue_before: np.ndarray,
    blue_after: np.ndarray,
    output_path: Path,
    shift_y: float,
    shift_x: float,
) -> None:
    qc_bin_size = max(
        1,
        int(
            np.ceil(
                max(green_total.shape) / REGISTRATION_QC_MAX_DIMENSION
            )
        ),
    )
    if qc_bin_size > 1:
        green_total = mean_bin_image(green_total, qc_bin_size)
        blue_before = mean_bin_image(blue_before, qc_bin_size)
        blue_after = mean_bin_image(blue_after, qc_bin_size)

    green_display = qc_display_image(green_total)
    before_display = qc_display_image(blue_before)
    after_display = qc_display_image(blue_after)
    before_rgb = np.stack(
        (before_display, green_display, before_display),
        axis=-1,
    )
    after_rgb = np.stack(
        (after_display, green_display, after_display),
        axis=-1,
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=200)
    axes[0].imshow(np.clip(before_rgb, 0, 1))
    axes[0].set_title("Before registration\ngreen vs blue (magenta)")
    axes[1].imshow(np.clip(after_rgb, 0, 1))
    axes[1].set_title(
        f"After translation registration\ndy={shift_y:+.2f}, dx={shift_x:+.2f} pixels"
    )
    for axis in axes:
        axis.set_axis_off()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def load_sp_raw_mosaics(
    mosaic_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, tuple[np.ndarray, ...]]]:
    rows, cols = parse_mosaic_shape(mosaic_dir.name)
    _, tile_paths = locate_flim_tiles(mosaic_dir)
    expected = rows * cols
    if len(tile_paths) != expected:
        raise ValueError(f"Expected {expected} Sp tiles, found {len(tile_paths)}")

    components: dict[int, tuple[np.ndarray, ...]] = {}
    red_tiles: dict[int, np.ndarray] = {}
    green_tiles: dict[int, np.ndarray] = {}
    blue_tiles: dict[int, np.ndarray] = {}
    for tile_number, path in tile_paths.items():
        red, green, blue = sp_components(read_flim_stack(path))
        components[tile_number] = (red, green, blue)
        red_tiles[tile_number] = red
        green_tiles[tile_number] = green
        blue_tiles[tile_number] = blue
    return (
        reconstruct_float_mosaic(red_tiles, rows, cols),
        reconstruct_float_mosaic(green_tiles, rows, cols),
        reconstruct_float_mosaic(blue_tiles, rows, cols),
        components,
    )


def process_sp_mosaic(mosaic_dir: Path, overwrite: bool) -> dict[str, Any]:
    print(f"    [Sp] {mosaic_dir.name}")
    output_dir = mosaic_dir / RGB_OUTPUT_SUBDIR
    output_path = output_dir / f"{mosaic_dir.name}_RGB_mosaic.png"
    metadata_path = output_dir / "rgb_metadata.json"
    if not overwrite and output_path.exists() and metadata_path.exists():
        print("      existing RGB found: skipped")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return {"status": "ok", "skipped_existing": True, **metadata}

    red, green, blue, tile_components = load_sp_raw_mosaics(mosaic_dir)
    rgb_mosaic, percentiles = build_rgb(red, green, blue)
    if overwrite or not output_path.exists():
        save_png(rgb_mosaic, output_path)

    # Preserve the old per-tile RGB outputs using mosaic-wide normalization.
    for tile_number, (tile_r, tile_g, tile_b) in tile_components.items():
        tile_rgb = np.stack(
            (
                normalize_channel(tile_r, *percentiles["R"]),
                normalize_channel(tile_g, *percentiles["G"]),
                normalize_channel(tile_b, *percentiles["B"]),
            ),
            axis=-1,
        )
        tile_uint8 = (np.clip(tile_rgb, 0, 1) * 255).round().astype(np.uint8)
        tile_output = output_dir / f"Im_{tile_number:05d}_RGB.png"
        if overwrite or not tile_output.exists():
            save_png(tile_uint8, tile_output)

    metadata = {
        "acquisition_type": "Sp_simultaneous",
        "mosaic": mosaic_dir.name,
        "input_directory": str(locate_flim_tiles(mosaic_dir)[0]),
        "output_rgb": str(output_path),
        "rgb_mapping": {
            "R": "sum corrected green bins 1-4",
            "G": "sum corrected green bins 5-16",
            "B": "sum corrected blue bins 1-16",
        },
        "percentiles": percentiles,
        "sequential_registration": False,
        "bin_size": 1,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    del red, green, blue, rgb_mosaic, tile_components
    gc.collect()
    return {"status": "ok", **metadata}


def load_sequential_raw_mosaics(
    green_dir: Path,
    blue_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    green_grid = parse_mosaic_shape(green_dir.name)
    blue_grid = parse_mosaic_shape(blue_dir.name)
    if green_grid != blue_grid:
        raise ValueError(f"A1/A0 grids differ: A1={green_grid}, A0={blue_grid}")
    rows, cols = green_grid

    _, green_paths = locate_flim_tiles(green_dir)
    _, blue_paths = locate_flim_tiles(blue_dir)
    if set(green_paths) != set(blue_paths):
        raise ValueError(
            "A1/A0 tile numbers differ: "
            f"A1={sorted(green_paths)}, A0={sorted(blue_paths)}"
        )
    expected = rows * cols
    if len(green_paths) != expected:
        raise ValueError(f"Expected {expected} A1/A0 tiles, found {len(green_paths)}")

    red_tiles: dict[int, np.ndarray] = {}
    green_tiles: dict[int, np.ndarray] = {}
    blue_tiles: dict[int, np.ndarray] = {}
    for tile_number in green_paths:
        red, green = a1_components(read_flim_stack(green_paths[tile_number]))
        blue = a0_component(read_flim_stack(blue_paths[tile_number]))
        if red.shape != blue.shape:
            raise ValueError(
                f"A1/A0 tile {tile_number} shapes differ: "
                f"A1={red.shape}, A0={blue.shape}"
            )
        red_tiles[tile_number] = red
        green_tiles[tile_number] = green
        blue_tiles[tile_number] = blue

    return (
        reconstruct_float_mosaic(red_tiles, rows, cols),
        reconstruct_float_mosaic(green_tiles, rows, cols),
        reconstruct_float_mosaic(blue_tiles, rows, cols),
    )


def load_a1_raw_mosaic(green_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = parse_mosaic_shape(green_dir.name)
    _, green_paths = locate_flim_tiles(green_dir)
    expected = rows * cols
    if len(green_paths) != expected:
        raise ValueError(f"Expected {expected} A1 tiles, found {len(green_paths)}")

    red_tiles: dict[int, np.ndarray] = {}
    green_tiles: dict[int, np.ndarray] = {}
    for tile_number, path in green_paths.items():
        red, green = a1_components(read_flim_stack(path))
        red_tiles[tile_number] = red
        green_tiles[tile_number] = green

    return (
        reconstruct_float_mosaic(red_tiles, rows, cols),
        reconstruct_float_mosaic(green_tiles, rows, cols),
    )


def process_a1_only_mosaic(
    green_dir: Path,
    bin_size: int,
    overwrite: bool,
) -> dict[str, Any]:
    print(f"    [A1 only] green={green_dir.name} | blue=unavailable")
    output_dir = green_dir / RGB_OUTPUT_SUBDIR
    output_path = output_dir / f"{green_dir.name}_green_only_binned_RGB_mosaic.png"
    metadata_path = output_dir / "rgb_metadata.json"
    if not overwrite and output_path.exists() and metadata_path.exists():
        print("      existing green-only RGB found: skipped")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return {"status": "ok", "skipped_existing": True, **metadata}

    red, green = load_a1_raw_mosaic(green_dir)
    red_binned = mean_bin_image(red, bin_size)
    green_binned = mean_bin_image(green, bin_size)
    blue_binned = np.zeros_like(green_binned, dtype=np.float32)
    rgb_mosaic, percentiles = build_rgb(red_binned, green_binned, blue_binned)
    save_png(rgb_mosaic, output_path)

    metadata = {
        "acquisition_type": "A1_green_only",
        "green_mosaic": green_dir.name,
        "blue_mosaic": None,
        "green_input_directory": str(locate_flim_tiles(green_dir)[0]),
        "blue_input_directory": None,
        "output_rgb": str(output_path),
        "rgb_mapping": {
            "R": "sum corrected A1 green bins 1-4",
            "G": "sum corrected A1 green bins 5-32",
            "B": "zero; A0 acquisition unavailable",
        },
        "visualization_only": True,
        "blue_channel_available": False,
        "registration_model": "none; A1 only",
        "binning": "non-overlapping block mean",
        "bin_size": bin_size,
        "original_shape": list(red.shape),
        "binned_shape": list(red_binned.shape),
        "percentiles": percentiles,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print("      blue channel set to zero (green-only visualization)")
    del red, green, red_binned, green_binned, blue_binned, rgb_mosaic
    gc.collect()
    return {"status": "ok", **metadata}


def process_sequential_pair(
    green_dir: Path,
    blue_dir: Path,
    bin_size: int,
    register: bool,
    overwrite: bool,
) -> dict[str, Any]:
    print(f"    [A1/A0] green={green_dir.name} | blue={blue_dir.name}")
    output_dir = green_dir / RGB_OUTPUT_SUBDIR
    output_path = output_dir / f"{green_dir.name}_registered_binned_RGB_mosaic.png"
    metadata_path = output_dir / "rgb_metadata.json"
    if not overwrite and output_path.exists() and metadata_path.exists():
        print("      existing RGB found: skipped")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return {"status": "ok", "skipped_existing": True, **metadata}

    red, green, blue = load_sequential_raw_mosaics(green_dir, blue_dir)
    green_total = red / SCALE_R + green / SCALE_G

    if register:
        registration = estimate_translation(green_total, blue)
        registered_blue = apply_translation(
            blue,
            shift_y=registration["shift_y_pixels"],
            shift_x=registration["shift_x_pixels"],
        )
    else:
        registration = {
            "shift_y_pixels": 0.0,
            "shift_x_pixels": 0.0,
            "registration_error": None,
            "phase_difference": None,
            "registration_downsample": None,
            "registration_upsample_factor": None,
        }
        registered_blue = blue

    qc_path = output_dir / f"{green_dir.name}_A1_A0_registration_QC.png"
    if overwrite or not qc_path.exists():
        save_registration_qc(
            green_total,
            blue,
            registered_blue,
            qc_path,
            shift_y=registration["shift_y_pixels"],
            shift_x=registration["shift_x_pixels"],
        )

    red_binned = mean_bin_image(red, bin_size)
    green_binned = mean_bin_image(green, bin_size)
    blue_binned = mean_bin_image(registered_blue, bin_size)
    rgb_mosaic, percentiles = build_rgb(red_binned, green_binned, blue_binned)
    if overwrite or not output_path.exists():
        save_png(rgb_mosaic, output_path)

    metadata = {
        "acquisition_type": "A1_A0_sequential",
        "green_mosaic": green_dir.name,
        "blue_mosaic": blue_dir.name,
        "green_input_directory": str(locate_flim_tiles(green_dir)[0]),
        "blue_input_directory": str(locate_flim_tiles(blue_dir)[0]),
        "output_rgb": str(output_path),
        "registration_qc": str(qc_path),
        "rgb_mapping": {
            "R": "sum corrected A1 green bins 1-4",
            "G": "sum corrected A1 green bins 5-32",
            "B": "sum corrected A0 blue bins 1-32",
        },
        "visualization_only": True,
        "registration_model": "global_translation" if register else "none",
        "binning": "non-overlapping block mean",
        "bin_size": bin_size,
        "original_shape": list(red.shape),
        "binned_shape": list(red_binned.shape),
        "percentiles": percentiles,
        **registration,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "      registration: "
        f"dy={registration['shift_y_pixels']:+.3f}, "
        f"dx={registration['shift_x_pixels']:+.3f} pixels"
    )
    del red, green, blue, green_total, registered_blue
    del red_binned, green_binned, blue_binned, rgb_mosaic
    gc.collect()
    return {"status": "ok", **metadata}


def process_patient(
    patient_dir: Path,
    bin_size: int,
    register: bool,
    overwrite: bool,
    continue_on_error: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    visits = sorted(
        (
            path
            for path in patient_dir.iterdir()
            if path.is_dir() and path.name.casefold().startswith("visit")
        ),
        key=natural_key,
    )
    print(f"\n[PATIENT] {patient_dir.name}: {len(visits)} visit(s)")

    for visit_dir in visits:
        print(f"  [VISIT] {visit_dir.name}")
        sp_dirs, sequential_pairs, green_only_dirs, warnings = (
            classify_visit_mosaics(visit_dir)
        )
        for warning in warnings:
            print(f"    [WARN] {warning}")

        jobs: list[tuple[str, Path, Path | None]] = [
            ("Sp", path, None) for path in sp_dirs
        ]
        jobs.extend(
            ("A1_A0", green, blue) for green, blue in sequential_pairs
        )
        jobs.extend(("A1_only", green, None) for green in green_only_dirs)

        for acquisition, first_dir, second_dir in jobs:
            base = {
                "patient": patient_dir.name,
                "visit": visit_dir.name,
                "acquisition_type": acquisition,
                "primary_mosaic": first_dir.name,
                "secondary_mosaic": second_dir.name if second_dir else "",
                "status": "error",
                "error": "",
            }
            try:
                if acquisition == "Sp":
                    result = process_sp_mosaic(first_dir, overwrite=overwrite)
                elif acquisition == "A1_A0":
                    if second_dir is None:
                        raise RuntimeError("Sequential job has no A0 directory")
                    result = process_sequential_pair(
                        first_dir,
                        second_dir,
                        bin_size=bin_size,
                        register=register,
                        overwrite=overwrite,
                    )
                elif acquisition == "A1_only":
                    result = process_a1_only_mosaic(
                        first_dir,
                        bin_size=bin_size,
                        overwrite=overwrite,
                    )
                else:
                    raise RuntimeError(f"Unknown acquisition job: {acquisition}")
                base.update(result)
            except Exception as exc:
                base["error"] = str(exc)
                print(f"    [ERROR] {first_dir.name}: {exc}")
                traceback.print_exc()
                if not continue_on_error:
                    raise
            rows.append(base)

    manifest_path = patient_dir / "rgb_generation_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    print(f"  Manifest: {manifest_path}")
    return rows


def validate_arguments(args: argparse.Namespace) -> None:
    if not args.data_root.expanduser().is_dir():
        raise NotADirectoryError(f"Data root not found: {args.data_root}")
    if args.sequential_bin_size <= 0:
        raise ValueError("--sequential-bin-size must be positive")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create multipatient pseudo-RGB mosaics from Sp and A1/A0 FLIM."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--patients",
        nargs="+",
        default=None,
        help="Patient folder names. If omitted, process every p<number> folder.",
    )
    parser.add_argument(
        "--sequential-bin-size",
        type=int,
        default=SEQUENTIAL_BIN_SIZE,
    )
    parser.add_argument("--no-register-sequential", action="store_true")
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()
    validate_arguments(args)

    data_root = args.data_root.expanduser().resolve()
    patients = find_patient_directories(data_root, args.patients)
    register = REGISTER_SEQUENTIAL and not args.no_register_sequential
    overwrite = OVERWRITE and not args.no_overwrite
    continue_on_error = CONTINUE_ON_ERROR and not args.stop_on_error

    print("=" * 78)
    print("MULTIPATIENT FLIM TO PSEUDO-RGB")
    print(f"Data root:            {data_root}")
    print(f"Patients:             {', '.join(path.name for path in patients)}")
    print(f"Sequential register:  {register}")
    print(
        "Sequential bin size:  "
        f"{args.sequential_bin_size}x{args.sequential_bin_size}"
    )
    print("Sp mapping:           R=green 1-4, G=green 5-16, B=blue 1-16")
    print("A1/A0 mapping:        R=A1 1-4, G=A1 5-32, B=A0 1-32")
    print("A1-only mapping:      R=A1 1-4, G=A1 5-32, B=0")
    print("=" * 78)

    all_rows: list[dict[str, Any]] = []
    for patient_dir in patients:
        all_rows.extend(
            process_patient(
                patient_dir,
                bin_size=args.sequential_bin_size,
                register=register,
                overwrite=overwrite,
                continue_on_error=continue_on_error,
            )
        )

    successful = sum(row.get("status") == "ok" for row in all_rows)
    print("\n" + "=" * 78)
    print(f"Finished: {successful}/{len(all_rows)} acquisition(s) successful")
    print("=" * 78)


if __name__ == "__main__":
    main()
