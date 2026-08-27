#!/usr/bin/env python3
"""Calculate and store corrected PhasorPy phasor maps for every raw FLIM tile.

Pipeline applied independently to every tile and channel::

    raw lifetime TIFF
      -> final-bin correction and channel split
      -> per-bin cubic 1200x1200 -> 1000x1000 -> 1200x1200 resampling
      -> phasorpy.phasor.phasor_from_signal
      -> save DC/mean, G, and S

No median filtering, intensity threshold, phase calibration, or spatial mosaic
concatenation is performed. One BigTIFF is written per acquisition with axes::

    (channel, tile, component, y, x) == CTZYX

Channel order is green then blue whenever both are available. Component order
is ``dc_mean, g, s``. Tiles remain an independent array axis; they are never
spatially concatenated. Therefore 4x4, 3x4, and other tile counts are retained
as 16, 12, or the actual number of discovered tiles. A two-channel 16-tile
1200x1200 output occupies about 553 MB as uncompressed float32.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from phasorpy import __version__ as phasorpy_version
from phasorpy.phasor import phasor_from_signal

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from calibration_by_blue.flim_preprocessing import (  # noqa: E402
    CorrectionJob,
    corrected_tile_channels,
    discover_jobs,
    grid_shape_from_mosaic_name,
    sanitize_filename,
    source_maps,
    spatially_resample_channels,
    validate_tile_numbers_for_grid,
)

AXES = "CTZYX"
COMPONENTS = ("dc_mean", "g", "s")
OUTPUT_SUFFIX = "_corrected_phasor.tiff"
DEFAULT_DATA_ROOT = Path("/Users/schutyb/Documents/balu_lab/dod/data_curated")
DEFAULT_PATIENTS = ("p427", "p437", "p439", "p449", "p476")
DEFAULT_DOWNSAMPLE_PIXELS = 200
MANIFEST_FIELDS = (
    "patient",
    "visit",
    "mosaic",
    "acquisition_type",
    "channels",
    "number_of_channels",
    "number_of_tiles",
    "grid_rows",
    "grid_columns",
    "height",
    "width",
    "shape",
    "dtype",
    "estimated_size_bytes",
    "actual_size_bytes",
    "output_tiff",
    "metadata_json",
    "status",
    "error",
)


def calculate_phasorpy_maps(
    decay: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return PhasorPy mean/DC, G, and S as float32 arrays."""
    mean, real, imag = phasor_from_signal(
        np.asarray(decay, dtype=np.float32),
        axis=-1,
        harmonic=1,
        use_fft=False,
        dtype=np.float32,
        normalize=True,
        num_threads=0,
    )
    return (
        np.asarray(mean, dtype=np.float32),
        np.asarray(real, dtype=np.float32),
        np.asarray(imag, dtype=np.float32),
    )


def output_paths(job: CorrectionJob, output_root: Path) -> tuple[Path, Path]:
    directory = output_root / job.patient / job.visit
    stem = sanitize_filename(job.mosaic)
    return (
        directory / f"{stem}{OUTPUT_SUFFIX}",
        directory / f"{stem}_corrected_phasor.json",
    )


def verify_free_space(
    output_root: Path,
    estimated_bytes: int,
    minimum_free_gb: float,
) -> None:
    free_bytes = shutil.disk_usage(output_root).free
    reserve_bytes = int(minimum_free_gb * 1024**3)
    if free_bytes - estimated_bytes < reserve_bytes:
        raise OSError(
            "Insufficient free disk space for the next corrected phasor: "
            f"free={free_bytes / 1024**3:.2f} GiB, "
            f"required_output={estimated_bytes / 1024**3:.2f} GiB, "
            f"required_reserve={minimum_free_gb:.2f} GiB"
        )


def write_job(
    job: CorrectionJob,
    output_root: Path,
    overwrite: bool,
    downsample_pixels: int,
    resample_workers: int,
    minimum_free_gb: float,
) -> dict[str, Any]:
    output_tiff, metadata_json = output_paths(job, output_root)
    declared_grid_shape = grid_shape_from_mosaic_name(job.mosaic)
    maps = source_maps(job)
    tile_numbers = sorted(next(iter(maps.values())))
    grid_shape = validate_tile_numbers_for_grid(job.mosaic, set(tile_numbers))
    output_tiff.parent.mkdir(parents=True, exist_ok=True)
    partial_tiff = output_tiff.with_name(f".{output_tiff.name}.partial")

    if output_tiff.exists() and not overwrite:
        print(f"    [SKIP] Exists: {output_tiff}")
        return {
            "patient": job.patient,
            "visit": job.visit,
            "mosaic": job.mosaic,
            "acquisition_type": job.acquisition_type,
            "channels": ";".join(job.channels),
            "grid_rows": "" if grid_shape is None else grid_shape[0],
            "grid_columns": "" if grid_shape is None else grid_shape[1],
            "output_tiff": str(output_tiff),
            "metadata_json": str(metadata_json),
            "actual_size_bytes": output_tiff.stat().st_size,
            "status": "skipped_existing",
            "error": "",
        }

    first_corrected, first_paths, first_original_bins = corrected_tile_channels(
        job,
        maps,
        tile_numbers[0],
    )
    first_corrected = spatially_resample_channels(
        first_corrected,
        downsample_pixels,
        True,
        resample_workers,
    )
    channels = job.channels
    if tuple(first_corrected) != channels:
        raise RuntimeError(
            f"Unexpected channel order {tuple(first_corrected)}; expected {channels}"
        )

    first_phasors = {
        channel: calculate_phasorpy_maps(first_corrected[channel])
        for channel in channels
    }
    first_shape = first_phasors[channels[0]][0].shape
    if len(first_shape) != 2:
        raise ValueError(f"Expected 2-D phasor maps; got {first_shape}")
    height, width = first_shape
    output_shape = (
        len(channels),
        len(tile_numbers),
        len(COMPONENTS),
        height,
        width,
    )
    estimated_bytes = int(np.prod(output_shape)) * np.dtype(np.float32).itemsize
    verify_free_space(output_root, estimated_bytes, minimum_free_gb)

    if partial_tiff.exists():
        partial_tiff.unlink()
    output = tifffile.memmap(
        partial_tiff,
        shape=output_shape,
        dtype=np.float32,
        bigtiff=estimated_bytes >= 2**32,
        photometric="minisblack",
        metadata={"axes": AXES},
    )

    source_records: list[dict[str, Any]] = []
    try:
        for tile_index, tile_number in enumerate(tile_numbers):
            if tile_index == 0:
                corrected = first_corrected
                paths = first_paths
                original_bins = first_original_bins
                phasors = first_phasors
            else:
                corrected, paths, original_bins = corrected_tile_channels(
                    job,
                    maps,
                    tile_number,
                )
                corrected = spatially_resample_channels(
                    corrected,
                    downsample_pixels,
                    True,
                    resample_workers,
                )
                phasors = {
                    channel: calculate_phasorpy_maps(corrected[channel])
                    for channel in channels
                }

            for channel_index, channel in enumerate(channels):
                channel_maps = phasors[channel]
                if any(array.shape != (height, width) for array in channel_maps):
                    raise ValueError(
                        f"Phasor shape mismatch at tile {tile_number}, {channel}"
                    )
                for component_index, array in enumerate(channel_maps):
                    output[
                        channel_index,
                        tile_index,
                        component_index,
                    ] = array
                source_records.append(
                    {
                        "channel": channel,
                        "channel_index": channel_index,
                        "tile_number": tile_number,
                        "tile_index": tile_index,
                        "source_tiff": str(paths[channel]),
                        "original_bins": original_bins[channel],
                        "corrected_bins": int(corrected[channel].shape[-1]),
                    }
                )
            output.flush()
            print(f"    tile {tile_number:02d} ({tile_index + 1}/{len(tile_numbers)})")
    except BaseException:
        del output
        if partial_tiff.exists():
            partial_tiff.unlink()
        raise

    del output
    if output_tiff.exists():
        output_tiff.unlink()
    partial_tiff.replace(output_tiff)

    verified = tifffile.memmap(output_tiff, mode="r", squeeze=False)
    stored_shape = tuple(int(value) for value in verified.shape)
    del verified
    if stored_shape != output_shape:
        raise RuntimeError(
            f"Stored TIFF shape mismatch: {stored_shape} != {output_shape}"
        )

    metadata = {
        "patient": job.patient,
        "visit": job.visit,
        "mosaic": job.mosaic,
        "acquisition_type": job.acquisition_type,
        "channels": list(channels),
        "channel_order": "green before blue whenever both exist",
        "components": list(COMPONENTS),
        "component_order": "dc_mean, g, s",
        "array_shape": list(output_shape),
        "array_axes": AXES,
        "axis_meaning": {
            "C": "channel",
            "T": "tile; tiles remain spatially independent",
            "Z": "phasor component: dc_mean, g, s",
            "Y": "tile row",
            "X": "tile column",
        },
        "tile_numbers": tile_numbers,
        "mosaic_declared_grid_shape": (
            None if declared_grid_shape is None else list(declared_grid_shape)
        ),
        "mosaic_grid_shape": None if grid_shape is None else list(grid_shape),
        "mosaic_grid_validation": (
            "not declared in folder name"
            if declared_grid_shape is None
            else (
                f"validated {grid_shape[0]}x{grid_shape[1]}"
                if grid_shape == declared_grid_shape
                else (
                    f"validated rectangular crop {grid_shape[0]}x{grid_shape[1]} "
                    f"within declared {declared_grid_shape[0]}x"
                    f"{declared_grid_shape[1]}; original tile numbers retained"
                )
            )
        ),
        "dtype": "float32",
        "phasor": {
            "library": "PhasorPy",
            "version": phasorpy_version,
            "function": "phasorpy.phasor.phasor_from_signal",
            "harmonic": 1,
            "normalized": True,
            "dc_plane": "PhasorPy mean intensity over decay bins",
        },
        "preprocessing": {
            "bin_correction": True,
            "spatial_resampling": (
                f"Per-bin {(height, width)} -> "
                f"{(height - downsample_pixels, width - downsample_pixels)} -> "
                f"{(height, width)}"
            ),
            "interpolation": "cubic B-spline order 3",
            "boundary_mode": "reflect",
            "temporal_bins_mixed": False,
            "median_filter": False,
            "intensity_threshold": False,
            "phase_calibration": False,
        },
        "source_records": source_records,
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "patient": job.patient,
        "visit": job.visit,
        "mosaic": job.mosaic,
        "acquisition_type": job.acquisition_type,
        "channels": ";".join(channels),
        "number_of_channels": len(channels),
        "number_of_tiles": len(tile_numbers),
        "grid_rows": "" if grid_shape is None else grid_shape[0],
        "grid_columns": "" if grid_shape is None else grid_shape[1],
        "height": height,
        "width": width,
        "shape": str(output_shape),
        "dtype": "float32",
        "estimated_size_bytes": estimated_bytes,
        "actual_size_bytes": output_tiff.stat().st_size,
        "output_tiff": str(output_tiff),
        "metadata_json": str(metadata_json),
        "status": "ok",
        "error": "",
    }


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=MANIFEST_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--patients",
        nargs="*",
        default=list(DEFAULT_PATIENTS),
        help="Patient folder names. Default: p427 p437 p439 p449 p476.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Default: DATA_ROOT/corrected_phasor",
    )
    parser.add_argument(
        "--downsample-pixels",
        type=int,
        default=DEFAULT_DOWNSAMPLE_PIXELS,
        help=(
            "Pixels removed from each spatial dimension before cubic upsampling. "
            "Default: 200, giving 1200 -> 1000 -> 1200."
        ),
    )
    parser.add_argument("--resample-workers", type=int, default=4)
    parser.add_argument(
        "--minimum-free-gb",
        type=float,
        default=5.0,
        help="Stop before a mosaic if less than this reserve would remain.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else args.data_root / "corrected_phasor"
    )
    if args.downsample_pixels < 1:
        parser.error("--downsample-pixels must be at least 1")
    if args.resample_workers < 1:
        parser.error("--resample-workers must be at least 1")
    if args.minimum_free_gb < 0:
        parser.error("--minimum-free-gb cannot be negative")
    return args


def main() -> int:
    args = parse_args()
    if not args.data_root.is_dir():
        raise NotADirectoryError(args.data_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    jobs, warnings = discover_jobs(args.data_root, args.patients)
    for warning in warnings:
        print(f"[WARN] {warning}")
    if not jobs:
        raise RuntimeError("No Sp, A1, or A0 flim acquisitions were found")

    rows: list[dict[str, Any]] = []
    manifest_path = args.output_root / "corrected_phasor_manifest.csv"
    for index, job in enumerate(jobs, start=1):
        print(
            f"[{index}/{len(jobs)}] {job.patient} | {job.visit} | "
            f"{job.mosaic} | {job.acquisition_type} | {job.channels}"
        )
        try:
            row = write_job(
                job,
                args.output_root,
                args.overwrite,
                args.downsample_pixels,
                args.resample_workers,
                args.minimum_free_gb,
            )
        except Exception as error:
            row = {
                "patient": job.patient,
                "visit": job.visit,
                "mosaic": job.mosaic,
                "acquisition_type": job.acquisition_type,
                "channels": ";".join(job.channels),
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
            }
            print(f"    [ERROR] {row['error']}")
            if args.stop_on_error:
                raise
        rows.append(row)
        write_manifest(manifest_path, rows)

    successful = sum(row["status"] in {"ok", "skipped_existing"} for row in rows)
    errors = sum(row["status"] == "error" for row in rows)
    print(f"\nCompleted: {successful} successful/skipped, {errors} errors")
    print(f"Corrected phasor root: {args.output_root}")
    print(f"Manifest: {manifest_path}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
