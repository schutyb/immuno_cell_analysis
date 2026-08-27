#!/usr/bin/env python3
# ruff: noqa: E402
"""Run the production pipeline incrementally for exact mosaic selections.

Existing corrected phasors, delta rows, calibrated TIFFs, and manifest entries
for all other mosaics are preserved. Each selection uses the form
``patient/visit/mosaic``.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Iterable

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from calibration_by_blue.calculate_corrected_phasor import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_DOWNSAMPLE_PIXELS,
    write_job,
)
from calibration_by_blue.calculate_corrected_phasor import (
    write_manifest as write_corrected_manifest,
)
from calibration_by_blue.calibrate_phasors import (  # noqa: E402
    DEFAULT_CALIBRATION_TOP_DC_PERCENT,
    DEFAULT_COMPRESSION_LEVEL,
    DEFAULT_DELTA_MODE_BIN_WIDTH_DEG,
    DEFAULT_FILTER_REPEAT,
    DEFAULT_FILTER_SIZE,
    DEFAULT_MINIMUM_FREE_GB,
    DEFAULT_PLOT_BINS,
    DEFAULT_PLOT_TOP_DC_PERCENT,
    DEFAULT_REFERENCE_BINS,
    DEFAULT_REFERENCE_REFINE_WINDOW,
    load_mosaic_modes,
    modes_for_job,
    natural_key,
    process_job,
)
from calibration_by_blue.calibrate_phasors import (
    discover_jobs as discover_phasor_jobs,
)
from calibration_by_blue.calibrate_phasors import (
    write_manifest as write_calibration_manifest,
)
from calibration_by_blue.estimate_mosaic_deltas import (  # noqa: E402
    MOSAIC_FIELDS,
    TILE_FIELDS,
    estimate_job,
    write_csv,
)
from calibration_by_blue.flim_preprocessing import (  # noqa: E402
    CorrectionJob,
    source_maps,
)
from calibration_by_blue.flim_preprocessing import (
    discover_jobs as discover_raw_jobs,
)

Selection = tuple[str, str, str]


def parse_selection(value: str) -> Selection:
    parts = value.split("/", 2)
    if len(parts) != 3 or any(not part for part in parts):
        raise argparse.ArgumentTypeError(
            "selection must use patient/visit/mosaic, for example "
            "p476/visit04/Mosaic02_4x4_FOV600_z080_32Sp"
        )
    return parts[0], parts[1], parts[2]


def job_key(job: Any) -> Selection:
    return str(job.patient), str(job.visit), str(job.mosaic)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def replace_selected_rows(
    existing: Iterable[dict[str, Any]],
    new_rows: Iterable[dict[str, Any]],
    selections: set[Selection],
) -> list[dict[str, Any]]:
    kept = [
        row
        for row in existing
        if (str(row["patient"]), str(row["visit"]), str(row["mosaic"]))
        not in selections
    ]
    combined = [*kept, *new_rows]
    combined.sort(
        key=lambda row: natural_key(
            "/".join(
                (
                    str(row.get("patient", "")),
                    str(row.get("visit", "")),
                    str(row.get("mosaic", "")),
                    str(row.get("channel", "")),
                    str(row.get("tile_number", "")),
                )
            )
        )
    )
    return combined


def atomic_write(path: Path, writer: Any, *args: Any) -> None:
    """Write a CSV beside its destination and replace it atomically."""
    partial = path.with_name(f".{path.name}.partial")
    partial.unlink(missing_ok=True)
    try:
        writer(partial, *args)
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def select_exact_jobs(
    jobs: Iterable[Any], selections: set[Selection], stage: str
) -> list[Any]:
    selected = [job for job in jobs if job_key(job) in selections]
    found = {job_key(job) for job in selected}
    missing = sorted(selections - found, key=lambda value: natural_key("/".join(value)))
    if missing:
        formatted = ", ".join("/".join(value) for value in missing)
        raise RuntimeError(f"{stage}: requested mosaics not found: {formatted}")
    return selected


def estimation_args(args: argparse.Namespace) -> Namespace:
    return Namespace(
        filter_size=args.filter_size,
        filter_repeat=args.filter_repeat,
        calibration_top_dc_percent=args.calibration_top_dc_percent,
        reference_bins=args.reference_bins,
        reference_refine_window=args.reference_refine_window,
        delta_mode_bin_width_deg=args.delta_mode_bin_width_deg,
    )


def calibration_args(args: argparse.Namespace) -> Namespace:
    return Namespace(
        overwrite=args.overwrite,
        filter_size=args.filter_size,
        filter_repeat=args.filter_repeat,
        plot_top_dc_percent=args.plot_top_dc_percent,
        plot_bins=args.plot_bins,
        dpi=args.dpi,
        compression="zlib",
        compression_level=args.compression_level,
        minimum_free_gb=args.minimum_free_gb,
    )


def process_raw_phasors(
    raw_jobs: list[CorrectionJob], args: argparse.Namespace
) -> None:
    manifest = args.corrected_root / "corrected_phasor_manifest.csv"
    new_rows: list[dict[str, Any]] = []
    for index, job in enumerate(raw_jobs, start=1):
        print(f"[1/3 {index}/{len(raw_jobs)}] raw phasor: {'/'.join(job_key(job))}")
        new_rows.append(
            write_job(
                job,
                args.corrected_root,
                args.overwrite,
                args.downsample_pixels,
                args.resample_workers,
                args.minimum_free_gb,
            )
        )
    rows = replace_selected_rows(read_csv(manifest), new_rows, set(args.selections))
    atomic_write(manifest, write_corrected_manifest, rows)


def process_deltas(phasor_jobs: list[Any], args: argparse.Namespace) -> Path:
    args.delta_root.mkdir(parents=True, exist_ok=True)
    tile_csv = args.delta_root / "tile_delta_phase.csv"
    mosaic_csv = args.delta_root / "mosaic_delta_phase.csv"
    new_tile_rows: list[dict[str, Any]] = []
    new_mosaic_rows: list[dict[str, Any]] = []
    estimate_args = estimation_args(args)
    for index, job in enumerate(phasor_jobs, start=1):
        print(f"[2/3 {index}/{len(phasor_jobs)}] delta mode: {'/'.join(job_key(job))}")
        tile_rows, mosaic_rows = estimate_job(job, estimate_args)
        new_tile_rows.extend(tile_rows)
        new_mosaic_rows.extend(mosaic_rows)
    selections = set(args.selections)
    all_tile_rows = replace_selected_rows(read_csv(tile_csv), new_tile_rows, selections)
    all_mosaic_rows = replace_selected_rows(
        read_csv(mosaic_csv), new_mosaic_rows, selections
    )
    atomic_write(tile_csv, write_csv, all_tile_rows, TILE_FIELDS)
    atomic_write(mosaic_csv, write_csv, all_mosaic_rows, MOSAIC_FIELDS)
    return mosaic_csv


def process_final_phasors(
    phasor_jobs: list[Any], mosaic_csv: Path, args: argparse.Namespace
) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = args.output_root / "calibration_manifest.csv"
    all_modes = load_mosaic_modes(mosaic_csv)
    process_args = calibration_args(args)
    new_rows: list[dict[str, Any]] = []
    for index, job in enumerate(phasor_jobs, start=1):
        print(
            f"[3/3 {index}/{len(phasor_jobs)}] final phasor: {'/'.join(job_key(job))}"
        )
        new_rows.append(
            process_job(
                job,
                modes_for_job(job, all_modes),
                mosaic_csv,
                args.output_root,
                process_args,
            )
        )
    rows = replace_selected_rows(read_csv(manifest), new_rows, set(args.selections))
    atomic_write(manifest, write_calibration_manifest, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selections", nargs="+", type=parse_selection)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--corrected-root", type=Path)
    parser.add_argument("--delta-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--downsample-pixels", type=int, default=DEFAULT_DOWNSAMPLE_PIXELS
    )
    parser.add_argument("--resample-workers", type=int, default=4)
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
    parser.add_argument(
        "--plot-top-dc-percent",
        type=float,
        default=DEFAULT_PLOT_TOP_DC_PERCENT,
    )
    parser.add_argument("--plot-bins", type=int, default=DEFAULT_PLOT_BINS)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--compression-level", type=int, default=DEFAULT_COMPRESSION_LEVEL
    )
    parser.add_argument(
        "--minimum-free-gb", type=float, default=DEFAULT_MINIMUM_FREE_GB
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.corrected_root = (
        args.corrected_root.expanduser().resolve()
        if args.corrected_root
        else args.data_root / "corrected_phasor"
    )
    args.delta_root = (
        args.delta_root.expanduser().resolve()
        if args.delta_root
        else args.data_root / "mosaic_delta_phase"
    )
    args.output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else args.data_root / "calibrated_filtered_phasor"
    )
    if len(args.selections) != len(set(args.selections)):
        parser.error("duplicate mosaic selection")
    if args.filter_size < 1 or args.filter_size % 2 == 0:
        parser.error("--filter-size must be a positive odd integer")
    if args.filter_repeat < 1 or args.resample_workers < 1:
        parser.error("repeat and worker counts must be positive")
    if not 0 < args.calibration_top_dc_percent <= 100:
        parser.error("--calibration-top-dc-percent must be in (0, 100]")
    if not 0 < args.plot_top_dc_percent <= 100:
        parser.error("--plot-top-dc-percent must be in (0, 100]")
    if not math.isfinite(args.delta_mode_bin_width_deg) or (
        args.delta_mode_bin_width_deg <= 0
    ):
        parser.error("--delta-mode-bin-width-deg must be positive")
    return args


def main() -> int:
    args = parse_args()
    selections = set(args.selections)
    patients = sorted({selection[0] for selection in selections}, key=natural_key)
    raw_jobs, warnings = discover_raw_jobs(args.data_root, patients)
    for warning in warnings:
        print(f"[WARN] {warning}")
    raw_jobs = select_exact_jobs(raw_jobs, selections, "raw discovery")
    tile_total = 0
    for job in raw_jobs:
        tile_total += len(next(iter(source_maps(job).values())))
    print(f"Selected mosaics: {len(raw_jobs)}; spatial tiles: {tile_total}")
    for job in raw_jobs:
        print(f"  {'/'.join(job_key(job))}")
    if args.dry_run:
        print("Dry run complete; no outputs changed.")
        return 0

    process_raw_phasors(raw_jobs, args)
    phasor_jobs = select_exact_jobs(
        discover_phasor_jobs(args.corrected_root, patients),
        selections,
        "corrected-phasor discovery",
    )
    mosaic_csv = process_deltas(phasor_jobs, args)
    process_final_phasors(phasor_jobs, mosaic_csv, args)
    print("Incremental pipeline complete; all non-selected mosaics were preserved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
