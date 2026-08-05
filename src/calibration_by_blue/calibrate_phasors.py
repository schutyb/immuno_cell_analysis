#!/usr/bin/env python3
"""Apply final blue-derived self-calibration from saved mosaic delta phases.

This is the third pipeline stage. Inputs are the unfiltered, unthresholded
phasors written by ``calculate_corrected_phasor.py`` and the per-mosaic delta
CSV written by ``estimate_mosaic_deltas.py``.

For each mosaic this script:

1. reads the previously estimated blue and green mosaic delta modes;
2. applies the blue mode to blue and applies blue +2.1 degrees to simultaneous
   split green, or blue +1.55 degrees to sequential A1/A0 green;
3. if no blue reference exists, uses the saved own-green mode estimated
   against its 3.5--0.1 ns segment;
4. rotates the original (unfiltered, unthresholded) G/S arrays and only then
   median-filters DC/G/S for the final TIFF. No DC threshold is saved;
5. makes one representative overlay plot from the saved TIFF, retaining the
   brightest 40% of DC independently in each tile for visualization only.

The output TIFF layout remains ``(channel, tile, component, y, x)`` with
green before blue and components ``dc_mean, g, s``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import matplotlib
import numpy as np
import tifffile
from phasorpy import __version__ as phasorpy_version
from phasorpy.filter import phasor_filter_median
from phasorpy.lifetime import phasor_from_lifetime

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LASER_FREQUENCY_MHZ = 80.0
LIFETIME_RANGES_NS = {
    "blue": (0.0, 3.5),
    # Order documents the agreed fallback direction: 3.5 ns -> 0.1 ns.
    "green": (3.5, 0.1),
}
GREEN_OFFSET_DEG = {"Sp": 2.1, "A1_A0": 1.55}

DEFAULT_DATA_ROOT = Path("/Users/schutyb/Documents/balu_lab/dod/data_curated")
DEFAULT_PATIENTS = ("p449", "p439", "p427")
DEFAULT_FILTER_SIZE = 7
DEFAULT_FILTER_REPEAT = 2
DEFAULT_CALIBRATION_TOP_DC_PERCENT = 35.0
DEFAULT_PLOT_TOP_DC_PERCENT = 40.0
DEFAULT_REFERENCE_BINS = 128
DEFAULT_REFERENCE_REFINE_WINDOW = 1
DEFAULT_DELTA_MODE_BIN_WIDTH_DEG = 1.0
DEFAULT_PLOT_BINS = 256
DEFAULT_COMPRESSION_LEVEL = 6
DEFAULT_MINIMUM_FREE_GB = 5.0

AXES = "CTZYX"
COMPONENTS = ("dc_mean", "g", "s")
OUTPUT_SUFFIX = "_calibrated_filtered_phasor.tiff"
METADATA_SUFFIX = "_calibrated_filtered_phasor.json"
CHANNEL_COLORS = {"green": "#35b779", "blue": "#2c7fb8"}

MANIFEST_FIELDS = (
    "patient",
    "visit",
    "mosaic",
    "acquisition_type",
    "channels",
    "blue_mode_deg",
    "green_own_mode_deg",
    "blue_applied_delta_deg",
    "green_applied_delta_deg",
    "green_method",
    "output_tiff",
    "metadata_json",
    "plot_png",
    "status",
    "error",
)

MODE_REQUIRED_FIELDS = {
    "patient",
    "visit",
    "mosaic",
    "acquisition_type",
    "channel",
    "tiles_successful",
    "delta_phase_mode_deg",
}
ModeKey = tuple[str, str, str, str, str]


@dataclass(frozen=True)
class PhasorJob:
    patient: str
    visit: str
    mosaic: str
    acquisition_type: str
    tiff_path: Path
    channels: tuple[str, ...]
    tile_numbers: tuple[int, ...]


def natural_key(value: str | Path) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", str(value))
    )


def sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "mosaic"


def discover_jobs(phasor_root: Path, patients: Iterable[str]) -> list[PhasorJob]:
    jobs: list[PhasorJob] = []
    for patient in patients:
        patient_dir = phasor_root / patient
        if not patient_dir.is_dir():
            print(f"[WARN] Corrected-phasor directory not found: {patient_dir}")
            continue
        for metadata_path in sorted(
            patient_dir.rglob("*_corrected_phasor.json"), key=natural_key
        ):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            channels = tuple(str(value) for value in metadata["channels"])
            components = tuple(str(value) for value in metadata["components"])
            tile_numbers = tuple(int(value) for value in metadata["tile_numbers"])
            if components != COMPONENTS:
                raise ValueError(
                    f"Unexpected component order in {metadata_path}: {components}"
                )
            if len(channels) == 2 and channels != ("green", "blue"):
                raise ValueError(
                    f"Two-channel data must be green then blue: {metadata_path}"
                )
            if not channels or any(c not in {"green", "blue"} for c in channels):
                raise ValueError(f"Invalid channels in {metadata_path}: {channels}")
            tiff_path = metadata_path.with_name(
                metadata_path.name.replace(
                    "_corrected_phasor.json", "_corrected_phasor.tiff"
                )
            )
            if not tiff_path.is_file():
                raise FileNotFoundError(tiff_path)
            jobs.append(
                PhasorJob(
                    patient=str(metadata["patient"]),
                    visit=str(metadata["visit"]),
                    mosaic=str(metadata["mosaic"]),
                    acquisition_type=str(metadata["acquisition_type"]),
                    tiff_path=tiff_path,
                    channels=channels,
                    tile_numbers=tile_numbers,
                )
            )
    jobs.sort(key=lambda job: natural_key(f"{job.patient}/{job.visit}/{job.mosaic}"))
    return jobs


def load_mosaic_modes(path: Path) -> dict[ModeKey, float]:
    """Load successful, unique channel delta modes from the estimation stage."""
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = MODE_REQUIRED_FIELDS.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                "Mosaic delta CSV is missing: " + ", ".join(sorted(missing))
            )
        modes: dict[ModeKey, float] = {}
        for row in reader:
            try:
                successful = int(float(row["tiles_successful"]))
                value = float(row["delta_phase_mode_deg"])
            except (TypeError, ValueError):
                continue
            if successful < 1 or not math.isfinite(value):
                continue
            channel = row["channel"].strip().casefold()
            if channel not in {"green", "blue"}:
                continue
            key = (
                row["patient"],
                row["visit"],
                row["mosaic"],
                row["acquisition_type"],
                channel,
            )
            if key in modes:
                raise ValueError(f"Duplicate mosaic delta row: {key}")
            modes[key] = value
    if not modes:
        raise ValueError(f"No successful mosaic delta modes in {path}")
    return modes


def modes_for_job(
    job: PhasorJob,
    all_modes: dict[ModeKey, float],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for channel in job.channels:
        key = (
            job.patient,
            job.visit,
            job.mosaic,
            job.acquisition_type,
            channel,
        )
        if key in all_modes:
            result[channel] = all_modes[key]
    return result


def median_filter_tile(
    mean: np.ndarray,
    real: np.ndarray,
    imag: np.ndarray,
    size: int,
    repeat: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    filtered = phasor_filter_median(
        np.asarray(mean, dtype=np.float32),
        np.asarray(real, dtype=np.float32),
        np.asarray(imag, dtype=np.float32),
        size=size,
        repeat=repeat,
        use_scipy=False,
        num_threads=0,
    )
    return tuple(np.asarray(v, dtype=np.float32) for v in filtered)  # type: ignore[return-value]


def brightest_dc_mask(
    mean: np.ndarray,
    real: np.ndarray,
    imag: np.ndarray,
    top_percent: float,
) -> tuple[np.ndarray, float, int]:
    """Return a mask retaining exactly the upper DC percentile conceptually."""
    valid = np.isfinite(mean) & np.isfinite(real) & np.isfinite(imag) & (mean > 0)
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        raise ValueError("No finite positive-DC phasor pixels")
    threshold = float(np.percentile(mean[valid], 100.0 - top_percent))
    mask = valid & (mean >= threshold)
    if not np.any(mask):
        raise ValueError("DC selection removed every pixel")
    return mask, threshold, valid_count


def modal_reference(
    real: np.ndarray,
    imag: np.ndarray,
    mask: np.ndarray,
    bins: int,
    refine_window: int,
) -> tuple[float, float]:
    g = np.asarray(real[mask], dtype=np.float64)
    s = np.asarray(imag[mask], dtype=np.float64)
    if g.size == 0:
        raise ValueError("No phasor points available for modal estimation")
    g_min, g_max = float(g.min()), float(g.max())
    s_min, s_max = float(s.min()), float(s.max())
    if math.isclose(g_min, g_max):
        g_min, g_max = g_min - 1e-6, g_max + 1e-6
    if math.isclose(s_min, s_max):
        s_min, s_max = s_min - 1e-6, s_max + 1e-6
    counts, g_edges, s_edges = np.histogram2d(
        g, s, bins=bins, range=((g_min, g_max), (s_min, s_max))
    )
    peak_g, peak_s = np.unravel_index(int(np.argmax(counts)), counts.shape)
    g_low = g_edges[max(0, peak_g - refine_window)]
    g_high = g_edges[min(bins, peak_g + refine_window + 1)]
    s_low = s_edges[max(0, peak_s - refine_window)]
    s_high = s_edges[min(bins, peak_s + refine_window + 1)]
    local = (g >= g_low) & (g <= g_high) & (s >= s_low) & (s <= s_high)
    if not np.any(local):
        return (
            float((g_edges[peak_g] + g_edges[peak_g + 1]) / 2),
            float((s_edges[peak_s] + s_edges[peak_s + 1]) / 2),
        )
    return float(np.mean(g[local])), float(np.mean(s[local]))


def theoretical_segment(
    channel: str,
) -> tuple[tuple[float, float], tuple[float, float]]:
    lifetimes = np.asarray(LIFETIME_RANGES_NS[channel], dtype=np.float64)
    real, imag = phasor_from_lifetime(LASER_FREQUENCY_MHZ, lifetimes)
    return (float(real[0]), float(imag[0])), (float(real[1]), float(imag[1]))


def phase_rotation_to_segment(
    reference_g: float,
    reference_s: float,
    channel: str,
) -> tuple[float, float, float]:
    """Rotate at constant modulation to the nearest point on the line segment."""
    (g1, s1), (g2, s2) = theoretical_segment(channel)
    dx, dy = g2 - g1, s2 - s1
    radius = math.hypot(reference_g, reference_s)
    if radius <= 0 or math.isclose(dx * dx + dy * dy, 0.0):
        raise ValueError("Degenerate reference point or calibration segment")

    # Intersections of P(t)=P1+t(P2-P1), t in [0,1], with the circle having
    # the measured modulation radius. Restricting t makes this a true segment.
    a = dx * dx + dy * dy
    b = 2.0 * (g1 * dx + s1 * dy)
    c = g1 * g1 + s1 * s1 - radius * radius
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0:
        raise ValueError("Measured modulation does not intersect calibration segment")
    root = math.sqrt(max(discriminant, 0.0))
    candidates: list[tuple[float, float, float]] = []
    for t in ((-b + root) / (2.0 * a), (-b - root) / (2.0 * a)):
        if -1e-9 <= t <= 1.0 + 1e-9:
            target_g = g1 + min(1.0, max(0.0, t)) * dx
            target_s = s1 + min(1.0, max(0.0, t)) * dy
            distance = math.hypot(target_g - reference_g, target_s - reference_s)
            candidates.append((distance, target_g, target_s))
    if not candidates:
        raise ValueError("Measured modulation intersects the line but not the segment")
    _, target_g, target_s = min(candidates, key=lambda value: value[0])
    measured_angle = math.atan2(reference_s, reference_g)
    target_angle = math.atan2(target_s, target_g)
    delta = math.atan2(
        math.sin(target_angle - measured_angle),
        math.cos(target_angle - measured_angle),
    )
    return math.degrees(delta), target_g, target_s


def delta_mode(values: Iterable[float], bin_width_deg: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("No successful tile delta phases")
    if array.size == 1 or np.allclose(array, array[0]):
        return float(np.mean(array))
    low = math.floor(float(array.min()) / bin_width_deg) * bin_width_deg
    high = math.ceil(float(array.max()) / bin_width_deg) * bin_width_deg
    edges = np.arange(low, high + bin_width_deg * 1.0001, bin_width_deg)
    if edges.size < 2:
        edges = np.asarray([low, low + bin_width_deg])
    counts, edges = np.histogram(array, bins=edges)
    index = int(np.argmax(counts))
    if index == len(counts) - 1:
        selected = (array >= edges[index]) & (array <= edges[index + 1])
    else:
        selected = (array >= edges[index]) & (array < edges[index + 1])
    return float(np.mean(array[selected]))


def rotate_phasor(
    real: np.ndarray, imag: np.ndarray, delta_phase_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    angle = math.radians(delta_phase_deg)
    cosine, sine = np.float32(math.cos(angle)), np.float32(math.sin(angle))
    real32 = np.asarray(real, dtype=np.float32)
    imag32 = np.asarray(imag, dtype=np.float32)
    return (
        np.asarray(real32 * cosine - imag32 * sine, dtype=np.float32),
        np.asarray(real32 * sine + imag32 * cosine, dtype=np.float32),
    )


def estimate_modes(
    source: np.ndarray,
    job: PhasorJob,
    args: argparse.Namespace,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    modes: dict[str, float] = {}
    records: list[dict[str, Any]] = []
    for channel_index, channel in enumerate(job.channels):
        successful: list[float] = []
        for tile_index, tile_number in enumerate(job.tile_numbers):
            record: dict[str, Any] = {
                "channel": channel,
                "tile_number": tile_number,
                "tile_index": tile_index,
            }
            try:
                mean_f, real_f, imag_f = median_filter_tile(
                    source[channel_index, tile_index, 0],
                    source[channel_index, tile_index, 1],
                    source[channel_index, tile_index, 2],
                    args.filter_size,
                    args.filter_repeat,
                )
                mask, threshold, valid_count = brightest_dc_mask(
                    mean_f,
                    real_f,
                    imag_f,
                    args.calibration_top_dc_percent,
                )
                reference_g, reference_s = modal_reference(
                    real_f,
                    imag_f,
                    mask,
                    args.reference_bins,
                    args.reference_refine_window,
                )
                delta, target_g, target_s = phase_rotation_to_segment(
                    reference_g, reference_s, channel
                )
                successful.append(delta)
                record.update(
                    status="ok",
                    dc_threshold=threshold,
                    valid_pixels=valid_count,
                    selected_pixels=int(np.count_nonzero(mask)),
                    reference_g=reference_g,
                    reference_s=reference_s,
                    target_g=target_g,
                    target_s=target_s,
                    delta_phase_deg=delta,
                )
            except Exception as error:
                record.update(status="error", error=f"{type(error).__name__}: {error}")
            records.append(record)
        if successful:
            modes[channel] = delta_mode(successful, args.delta_mode_bin_width_deg)
    return modes, records


def applied_deltas(
    job: PhasorJob, modes: dict[str, float]
) -> tuple[dict[str, float], dict[str, str]]:
    applied: dict[str, float] = {}
    methods: dict[str, str] = {}
    blue_mode = modes.get("blue")
    green_mode = modes.get("green")
    if "blue" in job.channels:
        if blue_mode is None:
            raise ValueError("Blue acquisition has no successful blue tile mode")
        applied["blue"] = blue_mode
        methods["blue"] = "blue_tile_delta_histogram_mode_to_0_3p5ns"
    if "green" in job.channels:
        if blue_mode is not None and job.acquisition_type in GREEN_OFFSET_DEG:
            offset = GREEN_OFFSET_DEG[job.acquisition_type]
            applied["green"] = blue_mode + offset
            methods["green"] = f"blue_mode_plus_{offset:g}deg_{job.acquisition_type}"
        elif green_mode is not None:
            applied["green"] = green_mode
            methods["green"] = "own_green_mode_to_3p5_0p1ns_no_blue"
        else:
            raise ValueError("Green acquisition has neither blue nor green mode")
    return applied, methods


def output_paths(job: PhasorJob, output_root: Path) -> tuple[Path, Path, Path]:
    directory = output_root / job.patient / job.visit
    stem = sanitize_filename(job.mosaic)
    return (
        directory / f"{stem}{OUTPUT_SUFFIX}",
        directory / f"{stem}{METADATA_SUFFIX}",
        directory / f"{stem}_phasor_overlay.png",
    )


def write_filtered_tiff(
    source: np.ndarray,
    job: PhasorJob,
    applied: dict[str, float],
    output_tiff: Path,
    args: argparse.Namespace,
) -> None:
    output_shape = tuple(int(value) for value in source.shape)
    partial = output_tiff.with_name(f".{output_tiff.name}.partial.tiff")
    estimated_bytes = int(np.prod(output_shape)) * np.dtype(np.float32).itemsize
    free_bytes = shutil.disk_usage(output_tiff.parent).free
    reserve = int(args.minimum_free_gb * 1024**3)
    if free_bytes - estimated_bytes < reserve:
        raise OSError(
            "Insufficient free space: "
            f"free={free_bytes / 1024**3:.2f} GiB, "
            f"estimated={estimated_bytes / 1024**3:.2f} GiB, "
            f"reserve={args.minimum_free_gb:.2f} GiB"
        )

    def planes() -> Iterator[np.ndarray]:
        for channel_index, channel in enumerate(job.channels):
            for tile_index, tile_number in enumerate(job.tile_numbers):
                print(f"    write {channel} tile {tile_number}")
                mean = source[channel_index, tile_index, 0]
                rotated_g, rotated_s = rotate_phasor(
                    source[channel_index, tile_index, 1],
                    source[channel_index, tile_index, 2],
                    applied[channel],
                )
                mean_f, real_f, imag_f = median_filter_tile(
                    mean,
                    rotated_g,
                    rotated_s,
                    args.filter_size,
                    args.filter_repeat,
                )
                # Deliberately no DC mask: final TIFF is filtered, not thresholded.
                yield mean_f
                yield real_f
                yield imag_f

    partial.unlink(missing_ok=True)
    compression = None if args.compression == "none" else args.compression
    compressionargs = (
        {"level": args.compression_level} if compression == "zlib" else None
    )
    try:
        tifffile.imwrite(
            partial,
            data=planes(),
            shape=output_shape,
            dtype=np.float32,
            bigtiff=True,
            photometric="minisblack",
            metadata={"axes": AXES},
            compression=compression,
            compressionargs=compressionargs,
        )
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    output_tiff.unlink(missing_ok=True)
    partial.replace(output_tiff)
    with tifffile.TiffFile(output_tiff) as tif:
        series = tif.series[0]
        stored_shape = tuple(int(value) for value in series.shape)
        stored_dtype = np.dtype(series.dtype)
        stored_pages = len(series.pages)
    # Tifffile may omit singleton dimensions from the series view.
    squeezed_shape = tuple(size for size in output_shape if size != 1)
    expected_pages = int(np.prod(output_shape[:3]))
    if (
        stored_shape not in {output_shape, squeezed_shape}
        or stored_dtype != np.dtype(np.float32)
        or stored_pages != expected_pages
    ):
        output_tiff.unlink(missing_ok=True)
        raise RuntimeError(
            "Stored TIFF mismatch: "
            f"shape={stored_shape}, dtype={stored_dtype}, pages={stored_pages}; "
            f"expected shape={output_shape}, dtype=float32, "
            f"pages={expected_pages}"
        )


def accumulate_plot_histograms(
    calibrated_tiff: Path,
    job: PhasorJob,
    top_percent: float,
    bins: int,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    histograms = {
        channel: np.zeros((bins, bins), dtype=np.uint64) for channel in job.channels
    }
    counts = {channel: 0 for channel in job.channels}
    with tifffile.TiffFile(calibrated_tiff) as tif:
        pages = tif.series[0].pages
        expected_pages = len(job.channels) * len(job.tile_numbers) * 3
        if len(pages) != expected_pages:
            raise ValueError(
                f"Expected {expected_pages} TIFF pages, found {len(pages)}"
            )
        for channel_index, channel in enumerate(job.channels):
            for tile_index in range(len(job.tile_numbers)):
                page = (channel_index * len(job.tile_numbers) + tile_index) * 3
                mean = pages[page].asarray()
                real = pages[page + 1].asarray()
                imag = pages[page + 2].asarray()
                mask, _, _ = brightest_dc_mask(mean, real, imag, top_percent)
                histogram, _, _ = np.histogram2d(
                    np.asarray(real[mask], dtype=np.float64),
                    np.asarray(imag[mask], dtype=np.float64),
                    bins=bins,
                    range=((0.0, 1.0), (0.0, 0.65)),
                )
                histograms[channel] += histogram.astype(np.uint64)
                counts[channel] += int(np.count_nonzero(mask))
    return histograms, counts


def plot_overlay(
    calibrated_tiff: Path,
    job: PhasorJob,
    applied: dict[str, float],
    output_png: Path,
    args: argparse.Namespace,
) -> None:
    histograms, counts = accumulate_plot_histograms(
        calibrated_tiff,
        job,
        args.plot_top_dc_percent,
        args.plot_bins,
    )
    figure, axis = plt.subplots(figsize=(8.5, 7.0))
    x_edges = np.linspace(0.0, 1.0, args.plot_bins + 1)
    y_edges = np.linspace(0.0, 0.65, args.plot_bins + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    for channel in job.channels:
        hist = histograms[channel].T
        positive = hist[hist > 0]
        if positive.size == 0:
            continue
        levels = np.unique(
            np.maximum(
                1,
                np.quantile(positive, [0.50, 0.75, 0.90, 0.97]).astype(int),
            )
        )
        if levels.size == 1:
            levels = np.asarray([levels[0], levels[0] + 1])
        axis.contour(
            x_centers,
            y_centers,
            hist,
            levels=levels,
            colors=CHANNEL_COLORS[channel],
            linewidths=np.linspace(0.8, 2.0, len(levels)),
        )
        start, end = theoretical_segment(channel)
        axis.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=CHANNEL_COLORS[channel],
            linestyle="--",
            linewidth=1.5,
            label=(f"{channel}: delta={applied[channel]:+.3f}°, n={counts[channel]:,}"),
        )
    theta = np.linspace(0.0, math.pi, 500)
    axis.plot(0.5 + 0.5 * np.cos(theta), 0.5 * np.sin(theta), color="black", lw=1)
    axis.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 0.65),
        xlabel="G",
        ylabel="S",
        aspect="equal",
        title=(
            f"{job.patient} | {job.visit} | {job.mosaic}\n"
            f"filtered TIFF; brightest {args.plot_top_dc_percent:g}% DC for plot only"
        ),
    )
    axis.grid(alpha=0.2)
    axis.legend(loc="upper right", fontsize=9)
    figure.tight_layout()
    figure.savefig(output_png, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)


def process_job(
    job: PhasorJob,
    modes: dict[str, float],
    delta_source_csv: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_tiff, metadata_json, plot_png = output_paths(job, output_root)
    output_tiff.parent.mkdir(parents=True, exist_ok=True)
    if output_tiff.exists() and metadata_json.exists() and not args.overwrite:
        print(f"    [SKIP] {output_tiff}")
        return {
            "patient": job.patient,
            "visit": job.visit,
            "mosaic": job.mosaic,
            "acquisition_type": job.acquisition_type,
            "channels": ";".join(job.channels),
            "output_tiff": output_tiff,
            "metadata_json": metadata_json,
            "plot_png": plot_png if plot_png.exists() else "",
            "status": "skipped_existing",
            "error": "",
        }
    source = tifffile.memmap(job.tiff_path, mode="r", squeeze=False)
    expected_prefix = (len(job.channels), len(job.tile_numbers), 3)
    if source.ndim != 5 or tuple(source.shape[:3]) != expected_prefix:
        actual = tuple(source.shape)
        del source
        raise ValueError(f"Expected {expected_prefix}+(Y,X), got {actual}")
    try:
        applied, methods = applied_deltas(job, modes)
        write_filtered_tiff(source, job, applied, output_tiff, args)
    finally:
        del source
    plot_overlay(output_tiff, job, applied, plot_png, args)
    metadata = {
        "pipeline": "final_blue_derived_self_calibration",
        "patient": job.patient,
        "visit": job.visit,
        "mosaic": job.mosaic,
        "acquisition_type": job.acquisition_type,
        "source_tiff": str(job.tiff_path),
        "output_tiff": str(output_tiff),
        "axes": AXES,
        "channels": list(job.channels),
        "components": list(COMPONENTS),
        "tile_numbers": list(job.tile_numbers),
        "laser_frequency_mhz": LASER_FREQUENCY_MHZ,
        "calibration": {
            "mosaic_delta_source_csv": str(delta_source_csv),
            "lifetime_ranges_ns": LIFETIME_RANGES_NS,
            "green_offsets_from_blue_deg": GREEN_OFFSET_DEG,
            "filter_size": args.filter_size,
            "filter_repeat": args.filter_repeat,
            "independent_mosaic_modes_deg": modes,
            "applied_delta_phase_deg": applied,
            "methods": methods,
        },
        "final_tiff": {
            "rotation_input": "original unfiltered and unthresholded G/S",
            "operation_order": "rotate G/S, then median-filter DC/G/S",
            "median_filter": {
                "library": "PhasorPy",
                "version": phasorpy_version,
                "size": args.filter_size,
                "repeat": args.filter_repeat,
                "tilewise": True,
            },
            "thresholded": False,
        },
        "representative_plot": {
            "source": "final calibrated filtered TIFF",
            "brightest_dc_percent": args.plot_top_dc_percent,
            "threshold_affects_tiff": False,
            "path": str(plot_png),
        },
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "patient": job.patient,
        "visit": job.visit,
        "mosaic": job.mosaic,
        "acquisition_type": job.acquisition_type,
        "channels": ";".join(job.channels),
        "blue_mode_deg": modes.get("blue", ""),
        "green_own_mode_deg": modes.get("green", ""),
        "blue_applied_delta_deg": applied.get("blue", ""),
        "green_applied_delta_deg": applied.get("green", ""),
        "green_method": methods.get("green", ""),
        "output_tiff": output_tiff,
        "metadata_json": metadata_json,
        "plot_png": plot_png,
        "status": "ok",
        "error": "",
    }


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
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
    parser.add_argument("--patients", nargs="+", default=list(DEFAULT_PATIENTS))
    parser.add_argument("--phasor-root", type=Path)
    parser.add_argument("--mosaic-delta-csv", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--filter-size", type=int, default=DEFAULT_FILTER_SIZE)
    parser.add_argument("--filter-repeat", type=int, default=DEFAULT_FILTER_REPEAT)
    parser.add_argument(
        "--plot-top-dc-percent", type=float, default=DEFAULT_PLOT_TOP_DC_PERCENT
    )
    parser.add_argument("--plot-bins", type=int, default=DEFAULT_PLOT_BINS)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--compression", choices=("zlib", "none"), default="zlib")
    parser.add_argument(
        "--compression-level", type=int, default=DEFAULT_COMPRESSION_LEVEL
    )
    parser.add_argument(
        "--minimum-free-gb",
        type=float,
        default=DEFAULT_MINIMUM_FREE_GB,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.phasor_root = (
        args.phasor_root.expanduser().resolve()
        if args.phasor_root
        else args.data_root / "corrected_phasor"
    )
    args.mosaic_delta_csv = (
        args.mosaic_delta_csv.expanduser().resolve()
        if args.mosaic_delta_csv
        else args.data_root / "mosaic_delta_phase" / "mosaic_delta_phase.csv"
    )
    args.output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else args.data_root / "calibrated_filtered_phasor"
    )
    if args.filter_size < 1 or args.filter_size % 2 == 0:
        parser.error("--filter-size must be a positive odd integer")
    if args.filter_repeat < 1:
        parser.error("--filter-repeat must be at least 1")
    if not 0 < args.plot_top_dc_percent <= 100:
        parser.error("--plot-top-dc-percent must be in (0, 100]")
    if args.plot_bins < 2:
        parser.error("--plot-bins must be at least 2")
    if args.dpi < 1 or args.minimum_free_gb < 0:
        parser.error("--dpi must be positive and --minimum-free-gb nonnegative")
    return args


def main() -> int:
    args = parse_args()
    if not args.phasor_root.is_dir():
        raise NotADirectoryError(args.phasor_root)
    if not args.mosaic_delta_csv.is_file():
        raise FileNotFoundError(args.mosaic_delta_csv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    all_modes = load_mosaic_modes(args.mosaic_delta_csv)
    jobs = discover_jobs(args.phasor_root, args.patients)
    if not jobs:
        raise RuntimeError("No resampled corrected-phasor mosaics were found")
    print(f"Mosaics: {len(jobs)}")
    print(f"Mosaic deltas: {args.mosaic_delta_csv}")
    print("Green: Sp blue+2.1 deg; A1/A0 blue+1.55 deg; own-green fallback")
    print("Final TIFF: calibrated + median-filtered, not thresholded")
    print(f"Plot: brightest {args.plot_top_dc_percent:g}% DC")
    rows: list[dict[str, Any]] = []
    manifest = args.output_root / "calibration_manifest.csv"
    errors = 0
    for index, job in enumerate(jobs, start=1):
        print(
            f"[{index}/{len(jobs)}] {job.patient} | {job.visit} | "
            f"{job.mosaic} | {job.acquisition_type}"
        )
        try:
            job_modes = modes_for_job(job, all_modes)
            row = process_job(
                job,
                job_modes,
                args.mosaic_delta_csv,
                args.output_root,
                args,
            )
        except Exception as error:
            errors += 1
            traceback.print_exc()
            row = {
                "patient": job.patient,
                "visit": job.visit,
                "mosaic": job.mosaic,
                "acquisition_type": job.acquisition_type,
                "channels": ";".join(job.channels),
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
            }
            if args.stop_on_error:
                rows.append(row)
                write_manifest(manifest, rows)
                raise
        rows.append(row)
        write_manifest(manifest, rows)
    print(f"Completed {len(jobs)} mosaics; errors={errors}")
    print(f"Manifest: {manifest}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
