#!/usr/bin/env python3
"""Estimate and persist tile and mosaic delta phases before calibration.

Input TIFFs are the resampled, unfiltered, unthresholded phasors created by
``calculate_corrected_phasor.py``. For each tile and channel this stage:

1. median-filters DC/G/S with a 7x7 kernel twice;
2. retains the brightest 35% of finite positive-DC pixels;
3. estimates the modal phasor coordinate;
4. calculates its phase-only rotation to the 0--3.5 ns blue segment or the
   3.5--0.1 ns green segment.

The histogram mode of successful tile rotations becomes the independent
channel delta for that mosaic. This stage never rotates or writes phasor TIFFs.
Its CSV is the required input to ``calibrate_phasors.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import traceback
from pathlib import Path
from typing import Any, Iterable

import tifffile

from calibration_by_blue.calibrate_phasors import (
    DEFAULT_CALIBRATION_TOP_DC_PERCENT,
    DEFAULT_DATA_ROOT,
    DEFAULT_DELTA_MODE_BIN_WIDTH_DEG,
    DEFAULT_FILTER_REPEAT,
    DEFAULT_FILTER_SIZE,
    DEFAULT_PATIENTS,
    DEFAULT_REFERENCE_BINS,
    DEFAULT_REFERENCE_REFINE_WINDOW,
    LIFETIME_RANGES_NS,
    PhasorJob,
    discover_jobs,
    estimate_modes,
)

TILE_FIELDS = (
    "patient",
    "visit",
    "mosaic",
    "acquisition_type",
    "channel",
    "tile_number",
    "tile_index",
    "status",
    "error",
    "source_phasor_tiff",
    "filter_size",
    "filter_repeat",
    "brightest_dc_percent",
    "dc_threshold",
    "valid_pixels",
    "selected_pixels",
    "reference_g",
    "reference_s",
    "target_g",
    "target_s",
    "delta_phase_deg",
)

MOSAIC_FIELDS = (
    "patient",
    "visit",
    "mosaic",
    "acquisition_type",
    "channel",
    "tiles_total",
    "tiles_successful",
    "tiles_failed",
    "delta_phase_mode_deg",
    "delta_mode_bin_width_deg",
    "source_phasor_tiff",
    "status",
)


def enrich_tile_records(
    job: PhasorJob,
    records: Iterable[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    common = {
        "patient": job.patient,
        "visit": job.visit,
        "mosaic": job.mosaic,
        "acquisition_type": job.acquisition_type,
        "source_phasor_tiff": str(job.tiff_path),
        "filter_size": args.filter_size,
        "filter_repeat": args.filter_repeat,
        "brightest_dc_percent": args.calibration_top_dc_percent,
    }
    return [{**common, **record} for record in records]


def mosaic_rows(
    job: PhasorJob,
    modes: dict[str, float],
    records: Iterable[dict[str, Any]],
    mode_bin_width_deg: float,
) -> list[dict[str, Any]]:
    records = list(records)
    rows: list[dict[str, Any]] = []
    for channel in job.channels:
        channel_records = [
            record for record in records if record.get("channel") == channel
        ]
        successful = [
            record for record in channel_records if record.get("status") == "ok"
        ]
        rows.append(
            {
                "patient": job.patient,
                "visit": job.visit,
                "mosaic": job.mosaic,
                "acquisition_type": job.acquisition_type,
                "channel": channel,
                "tiles_total": len(channel_records),
                "tiles_successful": len(successful),
                "tiles_failed": len(channel_records) - len(successful),
                "delta_phase_mode_deg": modes.get(channel, ""),
                "delta_mode_bin_width_deg": mode_bin_width_deg,
                "source_phasor_tiff": str(job.tiff_path),
                "status": "ok" if channel in modes else "error",
            }
        )
    return rows


def estimate_job(
    job: PhasorJob,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = tifffile.memmap(job.tiff_path, mode="r", squeeze=False)
    expected_prefix = (len(job.channels), len(job.tile_numbers), 3)
    if source.ndim != 5 or tuple(source.shape[:3]) != expected_prefix:
        actual = tuple(source.shape)
        del source
        raise ValueError(f"Expected {expected_prefix}+(Y,X), got {actual}")
    try:
        modes, raw_records = estimate_modes(source, job, args)
    finally:
        del source
    tile_rows = enrich_tile_records(job, raw_records, args)
    return tile_rows, mosaic_rows(
        job,
        modes,
        raw_records,
        args.delta_mode_bin_width_deg,
    )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: Iterable[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--patients", nargs="+", default=list(DEFAULT_PATIENTS))
    parser.add_argument("--phasor-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--filter-size", type=int, default=DEFAULT_FILTER_SIZE)
    parser.add_argument("--filter-repeat", type=int, default=DEFAULT_FILTER_REPEAT)
    parser.add_argument(
        "--calibration-top-dc-percent",
        type=float,
        default=DEFAULT_CALIBRATION_TOP_DC_PERCENT,
    )
    parser.add_argument("--reference-bins", type=int, default=DEFAULT_REFERENCE_BINS)
    parser.add_argument(
        "--reference-refine-window",
        type=int,
        default=DEFAULT_REFERENCE_REFINE_WINDOW,
    )
    parser.add_argument(
        "--delta-mode-bin-width-deg",
        type=float,
        default=DEFAULT_DELTA_MODE_BIN_WIDTH_DEG,
    )
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.phasor_root = (
        args.phasor_root.expanduser().resolve()
        if args.phasor_root
        else args.data_root / "corrected_phasor"
    )
    args.output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else args.data_root / "mosaic_delta_phase"
    )
    if args.filter_size < 1 or args.filter_size % 2 == 0:
        parser.error("--filter-size must be a positive odd integer")
    if args.filter_repeat < 1:
        parser.error("--filter-repeat must be positive")
    if not 0 < args.calibration_top_dc_percent <= 100:
        parser.error("--calibration-top-dc-percent must be in (0, 100]")
    if args.reference_bins < 2 or args.reference_refine_window < 0:
        parser.error("invalid reference histogram settings")
    if not math.isfinite(args.delta_mode_bin_width_deg) or (
        args.delta_mode_bin_width_deg <= 0
    ):
        parser.error("--delta-mode-bin-width-deg must be positive")
    return args


def main() -> int:
    args = parse_args()
    if not args.phasor_root.is_dir():
        raise NotADirectoryError(args.phasor_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    jobs = discover_jobs(args.phasor_root, args.patients)
    if not jobs:
        raise RuntimeError("No resampled corrected-phasor mosaics were found")
    tile_rows: list[dict[str, Any]] = []
    modes_rows: list[dict[str, Any]] = []
    errors = 0
    tile_csv = args.output_root / "tile_delta_phase.csv"
    mosaic_csv = args.output_root / "mosaic_delta_phase.csv"
    for index, job in enumerate(jobs, start=1):
        print(
            f"[{index}/{len(jobs)}] {job.patient} | {job.visit} | "
            f"{job.mosaic} | {job.acquisition_type}"
        )
        try:
            job_tiles, job_modes = estimate_job(job, args)
            tile_rows.extend(job_tiles)
            modes_rows.extend(job_modes)
        except Exception:
            errors += 1
            traceback.print_exc()
            if args.stop_on_error:
                raise
        write_csv(tile_csv, tile_rows, TILE_FIELDS)
        write_csv(mosaic_csv, modes_rows, MOSAIC_FIELDS)
    metadata = {
        "stage": "mosaic_delta_phase_estimation_before_self_calibration",
        "source_phasor_root": str(args.phasor_root),
        "patients": list(args.patients),
        "lifetime_ranges_ns": LIFETIME_RANGES_NS,
        "filter_size": args.filter_size,
        "filter_repeat": args.filter_repeat,
        "brightest_dc_percent": args.calibration_top_dc_percent,
        "reference_histogram_bins": args.reference_bins,
        "reference_refine_window": args.reference_refine_window,
        "delta_mode_bin_width_deg": args.delta_mode_bin_width_deg,
        "tile_csv": str(tile_csv),
        "mosaic_csv": str(mosaic_csv),
    }
    (args.output_root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Completed {len(jobs)} mosaics; errors={errors}")
    print(f"Tile deltas:   {tile_csv}")
    print(f"Mosaic deltas: {mosaic_csv}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
