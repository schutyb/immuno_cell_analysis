#!/usr/bin/env python3
"""Compare PhasorPy median filters after the production FLIM resampling.

By default this experiment processes one p449/visit01 mosaic and one
p449/visit04 mosaic. Each raw FLIM tile is corrected and split, every decay-bin
image is independently resampled with the production cubic 1200->1000->1200
method, and the first-harmonic phasor is calculated once. The same phasor is
then filtered with:

* 7x7 repeated twice (production reference)
* 9x9 repeated once
* 11x11 repeated once

Only the call to ``phasor_filter_median`` is timed. Plots retain the brightest
35% of finite positive-DC pixels independently per tile, matching calibration
estimation. Final CSVs report filter time and numerical similarity to 7x7x2.
No production TIFF or calibration result is modified.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from calibration_by_blue.calculate_corrected_phasor import (  # noqa: E402
    calculate_phasorpy_maps,
)
from calibration_by_blue.calibrate_phasors import (  # noqa: E402
    brightest_dc_mask,
    median_filter_tile,
    theoretical_segment,
)
from calibration_by_blue.flim_preprocessing import (  # noqa: E402
    CorrectionJob,
    corrected_tile_channels,
    discover_jobs,
    sanitize_filename,
    source_maps,
    spatially_resample_channels,
)

DEFAULT_DATA_ROOT = Path("/Users/schutyb/Documents/balu_lab/dod/data_curated")
DEFAULT_PATIENT = "p449"
DEFAULT_MOSAICS = {
    "visit01": "Mosaic03_4x4_FOV600_z110_32Sp",
    "visit04": "Mosaic04_4x4_FOV600_z135_32Sp",
}
DEFAULT_DOWNSAMPLE_PIXELS = 200
DEFAULT_RESAMPLE_WORKERS = 4
DEFAULT_TOP_DC_PERCENT = 35.0
DEFAULT_HISTOGRAM_BINS = 256
DEFAULT_DPI = 300
PHASOR_RANGE = ((0.0, 1.0), (0.0, 0.65))


@dataclass(frozen=True)
class FilterSpec:
    key: str
    size: int
    repeat: int

    @property
    def label(self) -> str:
        return f"{self.size}x{self.size} x{self.repeat}"


FILTERS = (
    FilterSpec("median7x2", 7, 2),
    FilterSpec("median9x1", 9, 1),
    FilterSpec("median11x1", 11, 1),
)
REFERENCE_FILTER_KEY = "median7x2"
CHANNEL_COLORS = {"green": "#35b779", "blue": "#2c7fb8"}

TIMING_FIELDS = (
    "patient",
    "visit",
    "mosaic",
    "channel",
    "filter",
    "filter_size",
    "filter_repeat",
    "tiles_processed",
    "filter_time_seconds",
    "seconds_per_tile",
    "speedup_vs_7x7x2",
    "selected_plot_pixels",
)

SIMILARITY_FIELDS = (
    "patient",
    "visit",
    "mosaic",
    "channel",
    "candidate_filter",
    "reference_filter",
    "pixels_compared",
    "g_rmse",
    "s_rmse",
    "phasor_distance_rmse",
    "mean_absolute_phase_difference_deg",
    "p95_absolute_phase_difference_deg",
)

MOSAIC_FIELDS = (
    "patient",
    "visit",
    "mosaic",
    "tiles_processed",
    "resampling_time_seconds",
    "phasor_calculation_time_seconds",
    "comparison_plot",
)


def select_jobs(
    data_root: Path,
    patient: str,
    visit01_mosaic: str,
    visit04_mosaic: str,
) -> list[CorrectionJob]:
    jobs, warnings = discover_jobs(data_root, [patient])
    for warning in warnings:
        print(f"[WARN] {warning}")
    requested = {
        ("visit01", visit01_mosaic),
        ("visit04", visit04_mosaic),
    }
    selected = [job for job in jobs if (job.visit, job.mosaic) in requested]
    found = {(job.visit, job.mosaic) for job in selected}
    missing = requested.difference(found)
    if missing:
        formatted = ", ".join(f"{visit}/{mosaic}" for visit, mosaic in missing)
        raise FileNotFoundError(f"Requested mosaic(s) not discovered: {formatted}")
    if any(job.acquisition_type != "Sp" for job in selected):
        raise ValueError("This comparison currently expects two-channel Sp mosaics")
    return sorted(selected, key=lambda job: (job.visit, job.mosaic))


def empty_histograms(bins: int, channels: tuple[str, ...]) -> dict[str, Any]:
    return {
        channel: {spec.key: np.zeros((bins, bins), dtype=np.uint64) for spec in FILTERS}
        for channel in channels
    }


def accumulate_histogram(
    histogram: np.ndarray,
    mean: np.ndarray,
    real: np.ndarray,
    imag: np.ndarray,
    top_dc_percent: float,
) -> int:
    mask, _, _ = brightest_dc_mask(mean, real, imag, top_dc_percent)
    tile_histogram, _, _ = np.histogram2d(
        np.asarray(real[mask], dtype=np.float64),
        np.asarray(imag[mask], dtype=np.float64),
        bins=histogram.shape[0],
        range=PHASOR_RANGE,
    )
    histogram += tile_histogram.astype(np.uint64)
    return int(np.count_nonzero(mask))


def phase_difference_degrees(
    reference_g: np.ndarray,
    reference_s: np.ndarray,
    candidate_g: np.ndarray,
    candidate_s: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    reference_phase = np.arctan2(reference_s[valid], reference_g[valid])
    candidate_phase = np.arctan2(candidate_s[valid], candidate_g[valid])
    wrapped = np.arctan2(
        np.sin(candidate_phase - reference_phase),
        np.cos(candidate_phase - reference_phase),
    )
    return np.abs(np.degrees(wrapped))


def initialize_similarity() -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for channel in ("green", "blue"):
        for spec in FILTERS:
            if spec.key == REFERENCE_FILTER_KEY:
                continue
            result[(channel, spec.key)] = {
                "n": 0,
                "sum_g_squared": 0.0,
                "sum_s_squared": 0.0,
                "sum_distance_squared": 0.0,
                "sum_abs_phase": 0.0,
                "phase_samples": [],
            }
    return result


def update_similarity(
    accumulator: dict[str, Any],
    reference: tuple[np.ndarray, np.ndarray, np.ndarray],
    candidate: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    reference_mean, reference_g, reference_s = reference
    candidate_mean, candidate_g, candidate_s = candidate
    valid = (
        np.isfinite(reference_mean)
        & np.isfinite(reference_g)
        & np.isfinite(reference_s)
        & np.isfinite(candidate_mean)
        & np.isfinite(candidate_g)
        & np.isfinite(candidate_s)
        & (reference_mean > 0)
        & (candidate_mean > 0)
    )
    count = int(np.count_nonzero(valid))
    if count == 0:
        return
    difference_g = np.asarray(candidate_g[valid] - reference_g[valid], np.float64)
    difference_s = np.asarray(candidate_s[valid] - reference_s[valid], np.float64)
    phase_difference = phase_difference_degrees(
        reference_g,
        reference_s,
        candidate_g,
        candidate_s,
        valid,
    )
    accumulator["n"] += count
    accumulator["sum_g_squared"] += float(np.sum(difference_g**2))
    accumulator["sum_s_squared"] += float(np.sum(difference_s**2))
    accumulator["sum_distance_squared"] += float(
        np.sum(difference_g**2 + difference_s**2)
    )
    accumulator["sum_abs_phase"] += float(np.sum(phase_difference))
    # An exact p95 across all points would retain tens of millions of values.
    # Keep a deterministic, bounded stride sample for this descriptive metric.
    stride = max(1, phase_difference.size // 100_000)
    accumulator["phase_samples"].append(phase_difference[::stride])


def finalize_similarity(
    job: CorrectionJob,
    accumulators: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (channel, candidate), values in accumulators.items():
        count = int(values["n"])
        if count == 0:
            continue
        samples = np.concatenate(values["phase_samples"])
        rows.append(
            {
                "patient": job.patient,
                "visit": job.visit,
                "mosaic": job.mosaic,
                "channel": channel,
                "candidate_filter": candidate,
                "reference_filter": REFERENCE_FILTER_KEY,
                "pixels_compared": count,
                "g_rmse": math.sqrt(values["sum_g_squared"] / count),
                "s_rmse": math.sqrt(values["sum_s_squared"] / count),
                "phasor_distance_rmse": math.sqrt(
                    values["sum_distance_squared"] / count
                ),
                "mean_absolute_phase_difference_deg": (values["sum_abs_phase"] / count),
                "p95_absolute_phase_difference_deg": float(np.percentile(samples, 95)),
            }
        )
    return rows


def histogram_image(histogram: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_less_equal(histogram.T, 0)


def draw_phasor_panel(
    axis: plt.Axes,
    histogram: np.ndarray,
    channel: str,
    title: str,
) -> None:
    image = histogram_image(histogram)
    positive = histogram[histogram > 0]
    if positive.size:
        axis.imshow(
            image,
            origin="lower",
            extent=(
                PHASOR_RANGE[0][0],
                PHASOR_RANGE[0][1],
                PHASOR_RANGE[1][0],
                PHASOR_RANGE[1][1],
            ),
            aspect="equal",
            interpolation="nearest",
            cmap="turbo",
            norm=LogNorm(vmin=1, vmax=max(2, int(positive.max()))),
        )
    theta = np.linspace(0.0, math.pi, 500)
    axis.plot(0.5 + 0.5 * np.cos(theta), 0.5 * np.sin(theta), color="black", lw=1)
    start, end = theoretical_segment(channel)
    axis.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color=CHANNEL_COLORS[channel],
        linestyle="--",
        linewidth=1.5,
    )
    axis.set(
        xlim=PHASOR_RANGE[0],
        ylim=PHASOR_RANGE[1],
        xlabel="G",
        ylabel="S",
        title=title,
    )
    axis.grid(alpha=0.15)


def save_plots(
    job: CorrectionJob,
    output_dir: Path,
    histograms: dict[str, Any],
    filter_times: dict[tuple[str, str], float],
    selected_counts: dict[tuple[str, str], int],
    tiles_processed: int,
    top_dc_percent: float,
    dpi: int,
) -> Path:
    stem = sanitize_filename(f"{job.patient}_{job.visit}_{job.mosaic}")
    channels = job.channels
    figure, axes = plt.subplots(
        len(channels),
        len(FILTERS),
        figsize=(5.2 * len(FILTERS), 4.8 * len(channels)),
        squeeze=False,
    )
    for row, channel in enumerate(channels):
        reference_time = filter_times[(channel, REFERENCE_FILTER_KEY)]
        for column, spec in enumerate(FILTERS):
            elapsed = filter_times[(channel, spec.key)]
            speedup = reference_time / elapsed if elapsed > 0 else float("nan")
            draw_phasor_panel(
                axes[row, column],
                histograms[channel][spec.key],
                channel,
                (
                    f"{channel.capitalize()} | median {spec.label}\n"
                    f"filter={elapsed:.3f} s total; {elapsed / tiles_processed:.3f} "
                    f"s/tile; speedup={speedup:.2f}x\n"
                    f"top {top_dc_percent:g}% DC; "
                    f"n={selected_counts[(channel, spec.key)]:,}"
                ),
            )
    figure.suptitle(
        f"{job.patient} | {job.visit} | {job.mosaic}\n"
        "Same resampled phasors; filter timing excludes I/O, resampling, and phasor",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    comparison_path = output_dir / f"{stem}_median_filter_comparison.png"
    figure.savefig(comparison_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    for spec in FILTERS:
        figure, axes = plt.subplots(
            1,
            len(channels),
            figsize=(6 * len(channels), 5.4),
            squeeze=False,
        )
        total_time = sum(filter_times[(channel, spec.key)] for channel in channels)
        for column, channel in enumerate(channels):
            elapsed = filter_times[(channel, spec.key)]
            draw_phasor_panel(
                axes[0, column],
                histograms[channel][spec.key],
                channel,
                (
                    f"{channel.capitalize()} | median {spec.label}\n"
                    f"filter time={elapsed:.3f} s ({elapsed / tiles_processed:.3f} "
                    "s/tile)"
                ),
            )
        figure.suptitle(
            f"{job.patient} | {job.visit} | {job.mosaic}\n"
            f"median {spec.label} | both-channel filter time={total_time:.3f} s",
            fontsize=13,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.92))
        figure.savefig(
            output_dir / f"{stem}_{spec.key}_phasor.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(figure)
    return comparison_path


def process_mosaic(
    job: CorrectionJob,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    maps = source_maps(job)
    tile_numbers = sorted(next(iter(maps.values())))
    if args.max_tiles is not None:
        tile_numbers = tile_numbers[: args.max_tiles]
    if not tile_numbers:
        raise ValueError(f"No tiles selected for {job.mosaic}")

    histograms = empty_histograms(args.histogram_bins, job.channels)
    filter_times = {
        (channel, spec.key): 0.0 for channel in job.channels for spec in FILTERS
    }
    selected_counts = {
        (channel, spec.key): 0 for channel in job.channels for spec in FILTERS
    }
    similarity = initialize_similarity()
    resampling_time = 0.0
    phasor_time = 0.0

    for tile_index, tile_number in enumerate(tile_numbers, start=1):
        print(f"    tile {tile_number} ({tile_index}/{len(tile_numbers)})")
        corrected, _, _ = corrected_tile_channels(job, maps, tile_number)
        start = time.perf_counter()
        resampled = spatially_resample_channels(
            corrected,
            args.downsample_pixels,
            True,
            args.resample_workers,
        )
        resampling_time += time.perf_counter() - start
        del corrected

        for channel in job.channels:
            start = time.perf_counter()
            mean, real, imag = calculate_phasorpy_maps(resampled[channel])
            phasor_time += time.perf_counter() - start
            filtered: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            for spec in FILTERS:
                start = time.perf_counter()
                filtered[spec.key] = median_filter_tile(
                    mean,
                    real,
                    imag,
                    spec.size,
                    spec.repeat,
                )
                elapsed = time.perf_counter() - start
                filter_times[(channel, spec.key)] += elapsed
                mean_f, real_f, imag_f = filtered[spec.key]
                selected_counts[(channel, spec.key)] += accumulate_histogram(
                    histograms[channel][spec.key],
                    mean_f,
                    real_f,
                    imag_f,
                    args.top_dc_percent,
                )

            reference = filtered[REFERENCE_FILTER_KEY]
            for spec in FILTERS:
                if spec.key == REFERENCE_FILTER_KEY:
                    continue
                update_similarity(
                    similarity[(channel, spec.key)],
                    reference,
                    filtered[spec.key],
                )
            del filtered, mean, real, imag
        del resampled

    comparison_plot = save_plots(
        job,
        output_dir,
        histograms,
        filter_times,
        selected_counts,
        len(tile_numbers),
        args.top_dc_percent,
        args.dpi,
    )
    timing_rows: list[dict[str, Any]] = []
    for channel in job.channels:
        reference_time = filter_times[(channel, REFERENCE_FILTER_KEY)]
        for spec in FILTERS:
            elapsed = filter_times[(channel, spec.key)]
            timing_rows.append(
                {
                    "patient": job.patient,
                    "visit": job.visit,
                    "mosaic": job.mosaic,
                    "channel": channel,
                    "filter": spec.key,
                    "filter_size": spec.size,
                    "filter_repeat": spec.repeat,
                    "tiles_processed": len(tile_numbers),
                    "filter_time_seconds": elapsed,
                    "seconds_per_tile": elapsed / len(tile_numbers),
                    "speedup_vs_7x7x2": (
                        reference_time / elapsed if elapsed > 0 else float("nan")
                    ),
                    "selected_plot_pixels": selected_counts[(channel, spec.key)],
                }
            )
    mosaic_row = {
        "patient": job.patient,
        "visit": job.visit,
        "mosaic": job.mosaic,
        "tiles_processed": len(tile_numbers),
        "resampling_time_seconds": resampling_time,
        "phasor_calculation_time_seconds": phasor_time,
        "comparison_plot": comparison_plot,
    }
    return timing_rows, finalize_similarity(job, similarity), mosaic_row


def write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--patient", default=DEFAULT_PATIENT)
    parser.add_argument("--visit01-mosaic", default=DEFAULT_MOSAICS["visit01"])
    parser.add_argument("--visit04-mosaic", default=DEFAULT_MOSAICS["visit04"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "aux" / "median_filter_comparison_results",
    )
    parser.add_argument(
        "--downsample-pixels", type=int, default=DEFAULT_DOWNSAMPLE_PIXELS
    )
    parser.add_argument(
        "--resample-workers", type=int, default=DEFAULT_RESAMPLE_WORKERS
    )
    parser.add_argument("--top-dc-percent", type=float, default=DEFAULT_TOP_DC_PERCENT)
    parser.add_argument("--histogram-bins", type=int, default=DEFAULT_HISTOGRAM_BINS)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument(
        "--max-tiles",
        type=int,
        help="Diagnostic only: process the first N tiles of each mosaic.",
    )
    args = parser.parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.downsample_pixels < 1 or args.resample_workers < 1:
        parser.error("resampling arguments must be positive")
    if not 0 < args.top_dc_percent <= 100:
        parser.error("--top-dc-percent must be in (0, 100]")
    if args.histogram_bins < 2 or args.dpi < 1:
        parser.error("histogram bins and DPI must be positive")
    if args.max_tiles is not None and args.max_tiles < 1:
        parser.error("--max-tiles must be positive")
    return args


def main() -> int:
    args = parse_args()
    if not args.data_root.is_dir():
        raise NotADirectoryError(args.data_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = select_jobs(
        args.data_root,
        args.patient,
        args.visit01_mosaic,
        args.visit04_mosaic,
    )
    timing_rows: list[dict[str, Any]] = []
    similarity_rows: list[dict[str, Any]] = []
    mosaic_rows: list[dict[str, Any]] = []
    for index, job in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {job.patient} | {job.visit} | {job.mosaic}")
        timings, similarities, mosaic = process_mosaic(job, args.output_dir, args)
        timing_rows.extend(timings)
        similarity_rows.extend(similarities)
        mosaic_rows.append(mosaic)
        write_csv(args.output_dir / "filter_timings.csv", timing_rows, TIMING_FIELDS)
        write_csv(
            args.output_dir / "filter_similarity_vs_7x7x2.csv",
            similarity_rows,
            SIMILARITY_FIELDS,
        )
        write_csv(
            args.output_dir / "mosaic_processing_times.csv",
            mosaic_rows,
            MOSAIC_FIELDS,
        )
    print(f"\nResults: {args.output_dir}")
    print("Timing: filter_timings.csv")
    print("Similarity: filter_similarity_vs_7x7x2.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
