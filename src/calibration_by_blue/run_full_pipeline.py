#!/usr/bin/env python3
"""Run the complete production FLIM phasor pipeline in three strict stages."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = SRC_ROOT.parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/immuno_cell_analysis_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/immuno_cell_analysis_cache")
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from calibration_by_blue.calculate_corrected_phasor import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_DOWNSAMPLE_PIXELS,
    DEFAULT_PATIENTS,
)
from calibration_by_blue.calibrate_phasors import (  # noqa: E402
    DEFAULT_CALIBRATION_TOP_DC_PERCENT,
    DEFAULT_COMPRESSION_LEVEL,
    DEFAULT_FILTER_REPEAT,
    DEFAULT_FILTER_SIZE,
    DEFAULT_PLOT_TOP_DC_PERCENT,
)
from calibration_by_blue.flim_preprocessing import (  # noqa: E402
    discover_jobs,
    source_maps,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--patients", nargs="+", default=list(DEFAULT_PATIENTS))
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
    parser.add_argument(
        "--plot-top-dc-percent",
        type=float,
        default=DEFAULT_PLOT_TOP_DC_PERCENT,
    )
    parser.add_argument(
        "--compression-level", type=int, default=DEFAULT_COMPRESSION_LEVEL
    )
    parser.add_argument("--minimum-free-gb", type=float, default=5.0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recalculate existing intermediate and final TIFFs.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and print commands only."
    )
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
    if args.downsample_pixels < 1 or args.resample_workers < 1:
        parser.error("resampling parameters must be positive")
    if args.filter_size < 1 or args.filter_size % 2 == 0:
        parser.error("--filter-size must be a positive odd integer")
    if args.filter_repeat < 1:
        parser.error("--filter-repeat must be positive")
    if not 0 < args.calibration_top_dc_percent <= 100:
        parser.error("--calibration-top-dc-percent must be in (0, 100]")
    if not 0 < args.plot_top_dc_percent <= 100:
        parser.error("--plot-top-dc-percent must be in (0, 100]")
    if not 0 <= args.compression_level <= 9:
        parser.error("--compression-level must be between 0 and 9")
    if args.minimum_free_gb < 0:
        parser.error("--minimum-free-gb cannot be negative")
    return args


def pipeline_commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    python = sys.executable
    scripts = Path(__file__).resolve().parent
    patients = ["--patients", *args.patients]
    overwrite = ["--overwrite"] if args.overwrite else []
    return [
        (
            "1/3 Resampling and raw phasors",
            [
                python,
                "-u",
                str(scripts / "calculate_corrected_phasor.py"),
                "--data-root",
                str(args.data_root),
                *patients,
                "--output-root",
                str(args.corrected_root),
                "--downsample-pixels",
                str(args.downsample_pixels),
                "--resample-workers",
                str(args.resample_workers),
                "--minimum-free-gb",
                str(args.minimum_free_gb),
                "--stop-on-error",
                *overwrite,
            ],
        ),
        (
            "2/3 Tile deltas and mosaic modes",
            [
                python,
                "-u",
                str(scripts / "estimate_mosaic_deltas.py"),
                "--data-root",
                str(args.data_root),
                *patients,
                "--phasor-root",
                str(args.corrected_root),
                "--output-root",
                str(args.delta_root),
                "--filter-size",
                str(args.filter_size),
                "--filter-repeat",
                str(args.filter_repeat),
                "--calibration-top-dc-percent",
                str(args.calibration_top_dc_percent),
                "--stop-on-error",
            ],
        ),
        (
            "3/3 Calibration and final filtered TIFFs",
            [
                python,
                "-u",
                str(scripts / "calibrate_phasors.py"),
                "--data-root",
                str(args.data_root),
                *patients,
                "--phasor-root",
                str(args.corrected_root),
                "--mosaic-delta-csv",
                str(args.delta_root / "mosaic_delta_phase.csv"),
                "--output-root",
                str(args.output_root),
                "--filter-size",
                str(args.filter_size),
                "--filter-repeat",
                str(args.filter_repeat),
                "--plot-top-dc-percent",
                str(args.plot_top_dc_percent),
                "--compression",
                "zlib",
                "--compression-level",
                str(args.compression_level),
                "--minimum-free-gb",
                str(args.minimum_free_gb),
                "--stop-on-error",
                *overwrite,
            ],
        ),
    ]


def print_command(command: list[str]) -> None:
    print(shlex.join(command), flush=True)


def main() -> int:
    args = parse_args()
    if not args.data_root.is_dir():
        raise NotADirectoryError(args.data_root)

    jobs, warnings = discover_jobs(args.data_root, args.patients)
    if not jobs:
        raise RuntimeError("No FLIM mosaics were discovered")
    spatial_tiles = 0
    channel_tiles = 0
    for job in jobs:
        maps = source_maps(job)
        tile_count = len(next(iter(maps.values())))
        spatial_tiles += tile_count
        channel_tiles += tile_count * len(job.channels)

    free_gib = shutil.disk_usage(args.data_root).free / 1024**3
    print("Production FLIM phasor pipeline", flush=True)
    print(f"Data root:       {args.data_root}", flush=True)
    print(f"Patients:        {', '.join(args.patients)}", flush=True)
    print(f"Mosaics:         {len(jobs)}", flush=True)
    print(f"Spatial tiles:   {spatial_tiles}", flush=True)
    print(f"Channel tiles:   {channel_tiles}", flush=True)
    print(f"Free disk:       {free_gib:.1f} GiB", flush=True)
    print(f"Raw phasors:     {args.corrected_root}", flush=True)
    print(f"Mosaic deltas:   {args.delta_root}", flush=True)
    print(f"Final products:  {args.output_root}", flush=True)
    print(
        "Calibration: blue 0-3.5 ns; green=blue+2.1 deg (Sp) or "
        "blue+1.55 deg (A1/A0); own-green 3.5-0.1 ns fallback",
        flush=True,
    )
    print(
        f"Filtering: {args.filter_size}x{args.filter_size}, "
        f"repeat={args.filter_repeat}; calibration DC top "
        f"{args.calibration_top_dc_percent:g}%; final TIFF unthresholded",
        flush=True,
    )
    for warning in warnings:
        print(f"[WARN] {warning}", flush=True)

    commands = pipeline_commands(args)
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(SRC_ROOT)
        if not existing_pythonpath
        else f"{SRC_ROOT}{os.pathsep}{existing_pythonpath}"
    )
    for label, command in commands:
        print(f"\n=== {label} ===", flush=True)
        print_command(command)
        if args.dry_run:
            continue
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=False,
        )
        if completed.returncode != 0:
            print(
                f"Pipeline stopped in stage {label} "
                f"(exit code {completed.returncode}).",
                file=sys.stderr,
            )
            return completed.returncode

    if args.dry_run:
        print("\nDry run complete; no processing was executed.", flush=True)
    else:
        print(f"\nPipeline complete. Final products: {args.output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
