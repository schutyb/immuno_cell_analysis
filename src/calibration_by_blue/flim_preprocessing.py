"""Shared raw-FLIM discovery, bin correction, and cubic resampling helpers."""

from __future__ import annotations

import concurrent.futures
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import zoom

from calibration_by_blue.flim_io import (
    classify_input_directories,
    detector_type_from_mosaic_name,
    extract_tile_number,
    find_flim_directories,
    find_tiles,
    normalize_single_detector_32_bins,
    prepare_tile,
    split_green_blue,
)


@dataclass(frozen=True)
class CorrectionJob:
    patient: str
    visit: str
    mosaic: str
    acquisition_type: str
    split_flim: Path | None = None
    green_flim: Path | None = None
    blue_flim: Path | None = None

    @property
    def channels(self) -> tuple[str, ...]:
        if self.split_flim is not None:
            return ("green", "blue")
        channels: list[str] = []
        if self.green_flim is not None:
            channels.append("green")
        if self.blue_flim is not None:
            channels.append("blue")
        return tuple(channels)


def natural_key(value: str | Path) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", str(value))
    )


def sanitize_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return value or "mosaic"


def visit_name(patient_dir: Path, flim_dir: Path) -> str:
    try:
        parts = flim_dir.relative_to(patient_dir).parts
    except ValueError:
        parts = flim_dir.parts
    for part in parts:
        if re.fullmatch(r"visit\d+", part, flags=re.IGNORECASE):
            return part
    return "visit_unknown"


def discover_jobs(
    data_root: Path,
    patients: list[str] | None,
) -> tuple[list[CorrectionJob], list[str]]:
    if patients:
        patient_dirs = [data_root / patient for patient in patients]
    else:
        patient_dirs = sorted(
            [
                path
                for path in data_root.iterdir()
                if path.is_dir() and re.fullmatch(r"p\d+", path.name, re.I)
            ],
            key=natural_key,
        )

    jobs: list[CorrectionJob] = []
    warnings: list[str] = []
    for patient_dir in patient_dirs:
        if not patient_dir.is_dir():
            warnings.append(f"Patient directory does not exist: {patient_dir}")
            continue

        discovered_flim_dirs = find_flim_directories(patient_dir)
        flim_dirs: list[Path] = []
        for flim_dir in discovered_flim_dirs:
            mosaic_name = flim_dir.parent.name
            if mosaic_name.casefold().startswith("out"):
                warnings.append(
                    f"{patient_dir.name}: ignored mosaic folder starting with "
                    f"'out': {flim_dir.parent}"
                )
                continue
            flim_dirs.append(flim_dir)
        split_dirs, pairs, discovery_warnings = classify_input_directories(flim_dirs)
        paired_dirs = {directory.resolve() for pair in pairs for directory in pair}

        for message in discovery_warnings:
            is_unpaired = (
                " sin A0 correspondiente" in message
                or " sin A1 correspondiente" in message
            )
            if is_unpaired:
                warnings.append(
                    f"{patient_dir.name}: {message} "
                    "(will be saved as a single-channel corrected phasor)"
                )
            else:
                warnings.append(f"{patient_dir.name}: {message}")

        for flim_dir in split_dirs:
            jobs.append(
                CorrectionJob(
                    patient=patient_dir.name,
                    visit=visit_name(patient_dir, flim_dir),
                    mosaic=flim_dir.parent.name,
                    acquisition_type="Sp",
                    split_flim=flim_dir,
                )
            )

        for green_flim, blue_flim in pairs:
            jobs.append(
                CorrectionJob(
                    patient=patient_dir.name,
                    visit=visit_name(patient_dir, green_flim),
                    mosaic=green_flim.parent.name,
                    acquisition_type="A1_A0",
                    green_flim=green_flim,
                    blue_flim=blue_flim,
                )
            )

        for flim_dir in flim_dirs:
            if flim_dir.resolve() in paired_dirs:
                continue
            detector = detector_type_from_mosaic_name(flim_dir.parent.name)
            if detector == "green":
                jobs.append(
                    CorrectionJob(
                        patient=patient_dir.name,
                        visit=visit_name(patient_dir, flim_dir),
                        mosaic=flim_dir.parent.name,
                        acquisition_type="A1_only",
                        green_flim=flim_dir,
                    )
                )
            elif detector == "blue":
                jobs.append(
                    CorrectionJob(
                        patient=patient_dir.name,
                        visit=visit_name(patient_dir, flim_dir),
                        mosaic=flim_dir.parent.name,
                        acquisition_type="A0_only",
                        blue_flim=flim_dir,
                    )
                )

    jobs.sort(key=lambda job: natural_key(f"{job.patient}/{job.visit}/{job.mosaic}"))
    return jobs, warnings


def numbered_tiles(flim_dir: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in find_tiles(flim_dir):
        number = extract_tile_number(path)
        if number is None:
            raise RuntimeError(f"Could not extract tile number: {path}")
        result[int(number)] = path
    if not result:
        raise FileNotFoundError(f"No Im_XXXXX TIFFs found in {flim_dir}")
    return result


def grid_shape_from_mosaic_name(name: str) -> tuple[int, int] | None:
    """Return any declared N x M acquisition grid, without fixed-size limits."""
    match = re.search(r"(?<!\d)(\d+)x(\d+)(?!\d)", name, flags=re.IGNORECASE)
    if match is None:
        return None
    rows, columns = int(match.group(1)), int(match.group(2))
    if rows < 1 or columns < 1:
        raise ValueError(f"Invalid mosaic grid in {name!r}: {rows}x{columns}")
    return rows, columns


def validate_tile_numbers_for_grid(
    mosaic: str,
    tile_numbers: set[int],
) -> tuple[int, int] | None:
    """Validate tiles and return their effective ``(rows, columns)`` grid.

    A complete declared grid is accepted as before. A rectangular crop is also
    valid when complete rows or columns were deliberately removed while the
    surviving TIFFs retained their original tile numbers. This preserves the
    acquisition provenance for cases such as tiles 5--16 from a declared 4x4
    scan, whose effective grid is 3 rows by 4 columns.
    """
    ordered = sorted(tile_numbers)
    if not ordered:
        raise ValueError(f"Mosaic {mosaic!r} contains no tiles")

    grid_shape = grid_shape_from_mosaic_name(mosaic)
    if grid_shape is None:
        expected_sequence = list(range(1, len(ordered) + 1))
        if ordered != expected_sequence:
            raise ValueError(
                f"Non-contiguous tile sequence for {mosaic}: "
                f"found={ordered}, expected={expected_sequence}"
            )
        return None

    rows, columns = grid_shape
    expected_count = rows * columns
    if ordered[0] < 1 or ordered[-1] > expected_count:
        raise ValueError(
            f"Mosaic {mosaic!r} declares {rows}x{columns}={expected_count} tiles "
            f"but contains out-of-grid numbers: {ordered}"
        )

    positions = {divmod(tile_number - 1, columns) for tile_number in ordered}
    active_rows = sorted({row for row, _ in positions})
    active_columns = sorted({column for _, column in positions})
    rectangular_positions = {
        (row, column) for row in active_rows for column in active_columns
    }
    rows_are_contiguous = active_rows == list(
        range(active_rows[0], active_rows[-1] + 1)
    )
    columns_are_contiguous = active_columns == list(
        range(active_columns[0], active_columns[-1] + 1)
    )
    if (
        positions != rectangular_positions
        or not rows_are_contiguous
        or not columns_are_contiguous
    ):
        raise ValueError(
            f"Mosaic {mosaic!r} declares {rows}x{columns}={expected_count} tiles "
            f"but {len(ordered)} were found and they do not form a rectangular "
            f"crop: {ordered}"
        )

    return len(active_rows), len(active_columns)


def source_maps(job: CorrectionJob) -> dict[str, dict[int, Path]]:
    if job.split_flim is not None:
        result = {"split": numbered_tiles(job.split_flim)}
    else:
        result = {}
        if job.green_flim is not None:
            result["green"] = numbered_tiles(job.green_flim)
        if job.blue_flim is not None:
            result["blue"] = numbered_tiles(job.blue_flim)
        if not result:
            raise RuntimeError("Correction job has no input channel")

    number_sets = [set(mapping) for mapping in result.values()]
    if any(numbers != number_sets[0] for numbers in number_sets[1:]):
        details = {channel: sorted(mapping) for channel, mapping in result.items()}
        raise ValueError(f"A1/A0 tile-number mismatch: {details}")
    validate_tile_numbers_for_grid(job.mosaic, number_sets[0])
    return result


def corrected_tile_channels(
    job: CorrectionJob,
    maps: dict[str, dict[int, Path]],
    tile_number: int,
) -> tuple[dict[str, np.ndarray], dict[str, Path], dict[str, int]]:
    """Return bin-corrected channel arrays in Y, X, bin order."""
    if "split" in maps:
        source = maps["split"][tile_number]
        raw, _, _ = prepare_tile(source)
        green, blue = split_green_blue(raw)
        return (
            {"green": green, "blue": blue},
            {"green": source, "blue": source},
            {"green": int(raw.shape[-1]), "blue": int(raw.shape[-1])},
        )

    corrected: dict[str, np.ndarray] = {}
    paths: dict[str, Path] = {}
    original_bins: dict[str, int] = {}
    for channel in job.channels:
        source = maps[channel][tile_number]
        raw, _, _ = prepare_tile(source)
        corrected[channel] = normalize_single_detector_32_bins(raw)
        paths[channel] = source
        original_bins[channel] = int(raw.shape[-1])
    return corrected, paths, original_bins


def cubic_downsample_upsample(
    tile: np.ndarray,
    pixels_to_remove: int,
    workers: int = 1,
) -> np.ndarray:
    """Cubic-resample each spatial lifetime plane without mixing bins."""
    tile = np.asarray(tile)
    if tile.ndim != 3:
        raise ValueError(f"Expected Y,X,bin tile; got {tile.shape}")
    height, width, bins = tile.shape
    target_height = height - pixels_to_remove
    target_width = width - pixels_to_remove
    if target_height < 2 or target_width < 2:
        raise ValueError(
            f"Cannot remove {pixels_to_remove} pixels from spatial shape "
            f"{(height, width)}"
        )

    def resample_plane(bin_index: int) -> tuple[int, np.ndarray]:
        plane = np.asarray(tile[..., bin_index], dtype=np.float32)
        down = zoom(
            plane,
            (target_height / height, target_width / width),
            output=np.float32,
            order=3,
            mode="reflect",
            prefilter=True,
            grid_mode=True,
        )
        if down.shape != (target_height, target_width):
            raise RuntimeError(
                f"Unexpected downsampled shape {down.shape}; expected "
                f"{(target_height, target_width)}"
            )
        up = zoom(
            down,
            (height / target_height, width / target_width),
            output=np.float32,
            order=3,
            mode="reflect",
            prefilter=True,
            grid_mode=True,
        )
        if up.shape != (height, width):
            raise RuntimeError(
                f"Unexpected upsampled shape {up.shape}; expected {(height, width)}"
            )
        finite = np.isfinite(plane)
        if np.any(finite):
            np.clip(
                up,
                float(np.min(plane[finite])),
                float(np.max(plane[finite])),
                out=up,
            )
        return bin_index, up

    result = np.empty((height, width, bins), dtype=np.float32)
    if workers == 1:
        for bin_index, plane in map(resample_plane, range(bins)):
            result[..., bin_index] = plane
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            for bin_index, plane in executor.map(resample_plane, range(bins)):
                result[..., bin_index] = plane

    if bins >= 2:
        result[..., -1] = result[..., -2]
    return result


def spatially_resample_channels(
    corrected: dict[str, np.ndarray],
    pixels_to_remove: int,
    enabled: bool,
    workers: int,
) -> dict[str, np.ndarray]:
    if not enabled:
        return corrected
    return {
        channel: cubic_downsample_upsample(tile, pixels_to_remove, workers)
        for channel, tile in corrected.items()
    }
