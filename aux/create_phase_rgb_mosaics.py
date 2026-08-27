#!/usr/bin/env python3
"""Create phase-colored RGB mosaics from final calibrated FLIM phasors.

For every calibrated mosaic, this visualization-only script:

1. reads the final median-filtered, non-thresholded DC/G/S TIFF;
2. colors green-channel phase with ``reds_to_greens``;
3. colors blue-channel phase with ``blues_to_greens``;
4. equalizes tile gain and normalizes DC once across the whole mosaic;
5. combines both phase colors with per-pixel normalized-DC weights;
6. saves full-resolution PNGs and one three-panel summary PNG; and
7. writes a PDF containing the three images for every mosaic.

No calibrated TIFF or calibration result is modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/immuno_cell_analysis_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/immuno_cell_analysis_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tifffile  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import PowerNorm  # noqa: E402
from PIL import Image  # noqa: E402

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from calibration_by_blue.calibrate_phasors import COMPONENTS  # noqa: E402
from calibration_by_blue.flim_preprocessing import (  # noqa: E402
    grid_shape_from_mosaic_name,
    validate_tile_numbers_for_grid,
)
from utils.color_scales import (  # noqa: E402
    get_phase_colormap,
    normalize_percentile,
    phase_to_rgb,
)

DEFAULT_INPUT_ROOT = Path(
    "/Users/schutyb/Documents/balu_lab/dod/data_curated/calibrated_filtered_phasor"
)
DEFAULT_PATIENTS = ("p427", "p437", "p439", "p449", "p476")
METADATA_SUFFIX = "_calibrated_filtered_phasor.json"
TIFF_SUFFIX = "_calibrated_filtered_phasor.tiff"
GREEN_SCALE = "reds_to_greens"
BLUE_SCALE = "blues_to_greens"

MANIFEST_FIELDS = (
    "patient",
    "visit",
    "mosaic",
    "acquisition_type",
    "channels",
    "tile_count",
    "mosaic_rows",
    "mosaic_columns",
    "tile_height",
    "tile_width",
    "phase_min_deg",
    "phase_max_deg",
    "phase_gamma",
    "normalization_mode",
    "intensity_pmin",
    "intensity_pmax",
    "intensity_gamma",
    "tile_reference_percentile",
    "tile_gain_minimum",
    "tile_gain_maximum",
    "green_applied_gain_minimum",
    "green_applied_gain_maximum",
    "green_global_dc_low",
    "green_global_dc_high",
    "blue_applied_gain_minimum",
    "blue_applied_gain_maximum",
    "blue_global_dc_low",
    "blue_global_dc_high",
    "green_png",
    "blue_png",
    "composite_png",
    "triptych_png",
    "source_tiff",
)


@dataclass(frozen=True)
class RgbJob:
    patient: str
    visit: str
    mosaic: str
    acquisition_type: str
    channels: tuple[str, ...]
    tile_numbers: tuple[int, ...]
    tiff_path: Path


@dataclass(frozen=True)
class OutputPaths:
    green: Path
    blue: Path
    composite: Path
    triptych: Path


@dataclass(frozen=True)
class IntensityNormalization:
    gains: tuple[float, ...]
    low: float
    high: float


def natural_key(value: str | Path) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", str(value))
    )


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "mosaic"


def discover_jobs(input_root: Path, patients: list[str]) -> list[RgbJob]:
    jobs: list[RgbJob] = []
    for patient in patients:
        patient_root = input_root / patient
        if not patient_root.is_dir():
            print(f"[WARN] Patient output not found: {patient_root}")
            continue
        for metadata_path in sorted(
            patient_root.rglob(f"*{METADATA_SUFFIX}"), key=natural_key
        ):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            channels = tuple(str(value) for value in metadata["channels"])
            tile_numbers = tuple(int(value) for value in metadata["tile_numbers"])
            if not channels or any(
                channel not in {"green", "blue"} for channel in channels
            ):
                raise ValueError(f"Unexpected channels in {metadata_path}: {channels}")
            tiff_path = metadata_path.with_name(
                metadata_path.name.removesuffix(METADATA_SUFFIX) + TIFF_SUFFIX
            )
            if not tiff_path.is_file():
                raise FileNotFoundError(tiff_path)
            jobs.append(
                RgbJob(
                    patient=str(metadata["patient"]),
                    visit=str(metadata["visit"]),
                    mosaic=str(metadata["mosaic"]),
                    acquisition_type=str(metadata["acquisition_type"]),
                    channels=channels,
                    tile_numbers=tile_numbers,
                    tiff_path=tiff_path,
                )
            )
    jobs.sort(key=lambda job: natural_key(f"{job.patient}/{job.visit}/{job.mosaic}"))
    return jobs


def compact_grid_positions(
    mosaic: str, tile_numbers: tuple[int, ...]
) -> tuple[dict[int, tuple[int, int]], tuple[int, int]]:
    """Map original tile numbers to a compact row-major rectangular grid."""
    declared = grid_shape_from_mosaic_name(mosaic)
    effective = validate_tile_numbers_for_grid(mosaic, set(tile_numbers))
    if declared is None:
        return (
            {tile: (0, index) for index, tile in enumerate(tile_numbers)},
            (1, len(tile_numbers)),
        )

    _, declared_columns = declared
    original_positions = {
        tile: divmod(tile - 1, declared_columns) for tile in tile_numbers
    }
    active_rows = sorted({row for row, _ in original_positions.values()})
    active_columns = sorted({column for _, column in original_positions.values()})
    row_lookup = {value: index for index, value in enumerate(active_rows)}
    column_lookup = {value: index for index, value in enumerate(active_columns)}
    positions = {
        tile: (row_lookup[row], column_lookup[column])
        for tile, (row, column) in original_positions.items()
    }
    shape = (len(active_rows), len(active_columns))
    if effective != shape:
        raise RuntimeError(
            f"Internal grid mismatch for {mosaic}: {effective=} versus {shape=}"
        )
    return positions, shape


def phase_color_and_weight(
    dc: np.ndarray,
    g: np.ndarray,
    s: np.ndarray,
    scale: str,
    intensity_gain: float,
    intensity_bounds: tuple[float, float] | None,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return phase color, normalized-DC weight, and display RGB for one tile."""
    dc = np.asarray(dc, dtype=np.float32)
    g = np.asarray(g, dtype=np.float32)
    s = np.asarray(s, dtype=np.float32)
    valid = np.isfinite(dc) & np.isfinite(g) & np.isfinite(s) & (dc > 0)
    phase_deg = np.full(g.shape, np.nan, dtype=np.float32)
    phase_deg[valid] = np.degrees(np.arctan2(s[valid], g[valid]))
    phase_color = phase_to_rgb(
        phase_deg,
        scale=scale,
        phase_min_deg=args.phase_min_deg,
        phase_max_deg=args.phase_max_deg,
        phase_gamma=args.phase_gamma,
    ).astype(np.float32, copy=False)
    intensity_for_scale = np.where(valid, dc * intensity_gain, np.nan)
    if intensity_bounds is None:
        weight = normalize_percentile(
            intensity_for_scale,
            pmin=args.intensity_pmin,
            pmax=args.intensity_pmax,
        )
    else:
        low, high = intensity_bounds
        weight = np.zeros(dc.shape, dtype=np.float32)
        weight[valid] = (intensity_for_scale[valid] - low) / (high - low)
        np.clip(weight, 0.0, 1.0, out=weight)
    if args.intensity_gamma != 1.0:
        weight = weight**args.intensity_gamma
    phase_color[~valid] = 0.0
    weight[~valid] = 0.0
    display_rgb = phase_color * weight[..., None]
    return phase_color, weight, display_rgb


def estimate_mosaic_intensity_normalization(
    pages: Any,
    channel_index: int,
    tile_count: int,
    args: argparse.Namespace,
) -> IntensityNormalization:
    """Estimate tile gains and one global DC range for an entire mosaic."""
    tile_samples: list[np.ndarray] = []
    tile_references: list[float] = []
    stride = args.normalization_sample_stride
    for tile_index in range(tile_count):
        page = (channel_index * tile_count + tile_index) * len(COMPONENTS)
        dc = np.asarray(pages[page].asarray(), dtype=np.float32)
        sample = dc[::stride, ::stride].ravel()
        sample = sample[np.isfinite(sample) & (sample > 0)]
        if sample.size == 0:
            raise ValueError(
                f"No positive finite DC values in channel {channel_index}, "
                f"tile index {tile_index}"
            )
        tile_samples.append(sample)
        tile_references.append(
            float(np.percentile(sample, args.tile_reference_percentile))
        )

    references = np.asarray(tile_references, dtype=np.float64)
    target = float(np.median(references))
    gains = np.clip(
        target / references,
        args.tile_gain_minimum,
        args.tile_gain_maximum,
    )
    corrected_samples = np.concatenate(
        [sample * gain for sample, gain in zip(tile_samples, gains, strict=True)]
    )
    low, high = np.percentile(
        corrected_samples,
        [args.intensity_pmin, args.intensity_pmax],
    )
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError(
            f"Invalid mosaic-wide DC normalization range: low={low}, high={high}"
        )
    return IntensityNormalization(
        gains=tuple(float(value) for value in gains),
        low=float(low),
        high=float(high),
    )


def combine_channels(
    green_color: np.ndarray,
    green_weight: np.ndarray,
    blue_color: np.ndarray,
    blue_weight: np.ndarray,
) -> np.ndarray:
    """Fuse phase colors using normalized DC as per-pixel confidence weights."""
    total_weight = green_weight + blue_weight
    weighted_color = (
        green_color * green_weight[..., None] + blue_color * blue_weight[..., None]
    )
    mixed_color = np.zeros_like(weighted_color, dtype=np.float32)
    np.divide(
        weighted_color,
        total_weight[..., None],
        out=mixed_color,
        where=total_weight[..., None] > 0,
    )
    union_brightness = 1.0 - (1.0 - green_weight) * (1.0 - blue_weight)
    return np.clip(mixed_color * union_brightness[..., None], 0.0, 1.0)


def to_uint8(rgb: np.ndarray) -> np.ndarray:
    return np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def output_paths(output_root: Path, job: RgbJob) -> OutputPaths:
    directory = output_root / job.patient / job.visit
    stem = sanitize_filename(job.mosaic)
    return OutputPaths(
        green=directory / f"{stem}_green_phase_rgb.png",
        blue=directory / f"{stem}_blue_phase_rgb.png",
        composite=directory / f"{stem}_dual_channel_phase_rgb.png",
        triptych=directory / f"{stem}_phase_rgb_triptych.png",
    )


def save_rgb_png(path: Path, rgb: np.ndarray, compress_level: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(
        path,
        format="PNG",
        compress_level=compress_level,
        optimize=False,
    )


def build_rgb_mosaics(
    job: RgbJob, args: argparse.Namespace
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    positions, (mosaic_rows, mosaic_columns) = compact_grid_positions(
        job.mosaic, job.tile_numbers
    )
    tile_count = len(job.tile_numbers)
    channel_lookup = {channel: index for index, channel in enumerate(job.channels)}

    with tifffile.TiffFile(job.tiff_path) as tif:
        series = tif.series[0]
        if np.dtype(series.dtype) != np.dtype(np.float32):
            raise ValueError(f"Expected float32, found {series.dtype}: {job.tiff_path}")
        pages = series.pages
        expected_pages = len(job.channels) * tile_count * len(COMPONENTS)
        if len(pages) != expected_pages:
            raise ValueError(
                f"Expected {expected_pages} pages, found {len(pages)}: {job.tiff_path}"
            )
        first_page = np.asarray(pages[0].asarray())
        if first_page.ndim != 2:
            raise ValueError(f"Expected 2D TIFF pages, found {first_page.shape}")
        tile_height, tile_width = first_page.shape
        mosaic_shape = (
            mosaic_rows * tile_height,
            mosaic_columns * tile_width,
            3,
        )
        green_mosaic = np.zeros(mosaic_shape, dtype=np.uint8)
        blue_mosaic = np.zeros(mosaic_shape, dtype=np.uint8)
        composite_mosaic = np.zeros(mosaic_shape, dtype=np.uint8)

        normalizations: dict[str, IntensityNormalization] = {}
        if args.normalization_mode == "mosaic":
            for channel, channel_index in channel_lookup.items():
                normalizations[channel] = estimate_mosaic_intensity_normalization(
                    pages,
                    channel_index,
                    tile_count,
                    args,
                )

        for tile_index, tile_number in enumerate(job.tile_numbers):
            row, column = positions[tile_number]
            y_slice = slice(row * tile_height, (row + 1) * tile_height)
            x_slice = slice(column * tile_width, (column + 1) * tile_width)
            channel_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            for channel, scale in (
                ("green", GREEN_SCALE),
                ("blue", BLUE_SCALE),
            ):
                if channel not in channel_lookup:
                    continue
                channel_index = channel_lookup[channel]
                page = (channel_index * tile_count + tile_index) * len(COMPONENTS)
                dc = pages[page].asarray()
                g = pages[page + 1].asarray()
                s = pages[page + 2].asarray()
                normalization = normalizations.get(channel)
                channel_data[channel] = phase_color_and_weight(
                    dc,
                    g,
                    s,
                    scale,
                    (
                        normalization.gains[tile_index]
                        if normalization is not None
                        else 1.0
                    ),
                    (
                        (normalization.low, normalization.high)
                        if normalization is not None
                        else None
                    ),
                    args,
                )

            empty_color = np.zeros((tile_height, tile_width, 3), dtype=np.float32)
            empty_weight = np.zeros((tile_height, tile_width), dtype=np.float32)
            green_color, green_weight, green_rgb = channel_data.get(
                "green", (empty_color, empty_weight, empty_color)
            )
            blue_color, blue_weight, blue_rgb = channel_data.get(
                "blue", (empty_color, empty_weight, empty_color)
            )
            composite_rgb = combine_channels(
                green_color,
                green_weight,
                blue_color,
                blue_weight,
            )
            green_mosaic[y_slice, x_slice] = to_uint8(green_rgb)
            blue_mosaic[y_slice, x_slice] = to_uint8(blue_rgb)
            composite_mosaic[y_slice, x_slice] = to_uint8(composite_rgb)

    normalization_details: dict[str, Any] = {}
    for channel in ("green", "blue"):
        normalization = normalizations.get(channel)
        normalization_details.update(
            {
                f"{channel}_applied_gain_minimum": (
                    min(normalization.gains) if normalization is not None else ""
                ),
                f"{channel}_applied_gain_maximum": (
                    max(normalization.gains) if normalization is not None else ""
                ),
                f"{channel}_global_dc_low": (
                    normalization.low if normalization is not None else ""
                ),
                f"{channel}_global_dc_high": (
                    normalization.high if normalization is not None else ""
                ),
            }
        )

    return (
        green_mosaic,
        blue_mosaic,
        composite_mosaic,
        {
            "mosaic_rows": mosaic_rows,
            "mosaic_columns": mosaic_columns,
            "tile_height": tile_height,
            "tile_width": tile_width,
            **normalization_details,
        },
    )


def make_triptych(
    job: RgbJob,
    green_rgb: np.ndarray,
    blue_rgb: np.ndarray,
    composite_rgb: np.ndarray,
    args: argparse.Namespace,
) -> plt.Figure:
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(18.0, 6.5),
        constrained_layout=True,
        facecolor="white",
    )
    panels = (
        (green_rgb, "Green phase RGB\nreds_to_greens", GREEN_SCALE),
        (blue_rgb, "Blue phase RGB\nblues_to_greens", BLUE_SCALE),
        (
            composite_rgb,
            "Dual-channel phase RGB\nDC-weighted color mixture",
            None,
        ),
    )
    norm = PowerNorm(
        gamma=args.phase_gamma,
        vmin=args.phase_min_deg,
        vmax=args.phase_max_deg,
    )
    for axis, (image, title, scale) in zip(axes, panels, strict=True):
        axis.imshow(image)
        axis.set_title(title, fontsize=12, fontweight="bold")
        axis.set_axis_off()
        if scale is not None:
            colorbar = figure.colorbar(
                ScalarMappable(norm=norm, cmap=get_phase_colormap(scale)),
                ax=axis,
                orientation="horizontal",
                fraction=0.045,
                pad=0.025,
                aspect=35,
            )
            colorbar.set_label("Phasor phase (degrees)")
    if "blue" not in job.channels:
        axes[1].text(
            0.5,
            0.5,
            "Blue channel unavailable",
            transform=axes[1].transAxes,
            ha="center",
            va="center",
            fontsize=15,
            color="white",
            bbox={"facecolor": "#333333", "edgecolor": "white", "alpha": 0.9},
        )
        axes[2].set_title(
            "Composite fallback\ngreen channel only",
            fontsize=12,
            fontweight="bold",
        )
    figure.suptitle(
        f"{job.patient} | {job.visit} | {job.mosaic} | {job.acquisition_type}\n"
        "Calibrated G/S | median 7x7 twice | no DC threshold | "
        + (
            "mosaic-wide DC normalization"
            if args.normalization_mode == "mosaic"
            else "per-tile DC normalization"
        ),
        fontsize=15,
        fontweight="bold",
    )
    return figure


def make_cover(jobs: list[RgbJob], args: argparse.Namespace) -> plt.Figure:
    patients = sorted({job.patient for job in jobs}, key=natural_key)
    visits = sorted({(job.patient, job.visit) for job in jobs}, key=natural_key)
    figure = plt.figure(figsize=(11.69, 8.27), constrained_layout=True)
    axis = figure.subplots()
    axis.axis("off")
    axis.text(
        0.5,
        0.75,
        "Calibrated FLIM phase RGB mosaics",
        ha="center",
        va="center",
        fontsize=28,
        fontweight="bold",
    )
    axis.text(
        0.5,
        0.60,
        "Green, blue, and DC-weighted dual-channel composition",
        ha="center",
        va="center",
        fontsize=18,
    )
    axis.text(
        0.5,
        0.39,
        (
            f"Patients: {', '.join(patients)}\n"
            f"Patient-visits: {len(visits)} | Mosaics: {len(jobs)}\n"
            f"Phase range: {args.phase_min_deg:g}-{args.phase_max_deg:g} degrees | "
            f"phase gamma: {args.phase_gamma:g}\n"
            f"Green scale: {GREEN_SCALE} | Blue scale: {BLUE_SCALE}\n"
            f"DC normalization: {args.normalization_mode} | percentiles "
            f"{args.intensity_pmin:g}-{args.intensity_pmax:g} | "
            f"gamma: {args.intensity_gamma:g}\n"
            f"Tile gain reference: percentile "
            f"{args.tile_reference_percentile:g} | gain limits "
            f"{args.tile_gain_minimum:g}-{args.tile_gain_maximum:g}"
        ),
        ha="center",
        va="center",
        fontsize=14,
        linespacing=1.55,
    )
    axis.text(
        0.5,
        0.15,
        (
            "Tile gain is equalized before one mosaic-wide DC scaling. The "
            "composite mixes phase colors in proportion to normalized DC. "
            "Source TIFFs remain unchanged."
        ),
        ha="center",
        va="center",
        fontsize=11,
        color="#444444",
        wrap=True,
    )
    return figure


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--patients", nargs="+", default=list(DEFAULT_PATIENTS))
    parser.add_argument("--visits", nargs="+")
    parser.add_argument("--mosaics", nargs="+")
    parser.add_argument("--phase-min-deg", type=float, default=0.0)
    parser.add_argument("--phase-max-deg", type=float, default=55.0)
    parser.add_argument("--phase-gamma", type=float, default=0.6)
    parser.add_argument(
        "--normalization-mode",
        choices=("mosaic", "tile"),
        default="mosaic",
        help=(
            "Use one mosaic-wide DC range after robust tile-gain equalization "
            "or preserve the original independent per-tile scaling."
        ),
    )
    parser.add_argument("--intensity-pmin", type=float, default=1.0)
    parser.add_argument("--intensity-pmax", type=float, default=99.0)
    parser.add_argument("--intensity-gamma", type=float, default=0.7)
    parser.add_argument("--tile-reference-percentile", type=float, default=50.0)
    parser.add_argument("--tile-gain-minimum", type=float, default=0.5)
    parser.add_argument("--tile-gain-maximum", type=float, default=2.0)
    parser.add_argument("--normalization-sample-stride", type=int, default=4)
    parser.add_argument("--png-compress-level", type=int, default=6)
    parser.add_argument("--triptych-dpi", type=int, default=180)
    parser.add_argument("--pdf-dpi", type=int, default=150)
    parser.add_argument("--max-mosaics", type=int)
    args = parser.parse_args()
    args.input_root = args.input_root.expanduser().resolve()
    args.output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else args.input_root / "phase_rgb_mosaics"
    )
    if args.phase_max_deg <= args.phase_min_deg:
        parser.error("--phase-max-deg must be greater than --phase-min-deg")
    if args.phase_gamma <= 0 or args.intensity_gamma <= 0:
        parser.error("gamma values must be positive")
    if not 0 <= args.intensity_pmin < args.intensity_pmax <= 100:
        parser.error("intensity percentiles must satisfy 0 <= pmin < pmax <= 100")
    if not 0 <= args.tile_reference_percentile <= 100:
        parser.error("--tile-reference-percentile must be in [0, 100]")
    if not 0 < args.tile_gain_minimum <= args.tile_gain_maximum:
        parser.error("tile gain limits must satisfy 0 < minimum <= maximum")
    if args.normalization_sample_stride < 1:
        parser.error("--normalization-sample-stride must be positive")
    if not 0 <= args.png_compress_level <= 9:
        parser.error("--png-compress-level must be in [0, 9]")
    if args.triptych_dpi < 1 or args.pdf_dpi < 1:
        parser.error("DPI values must be positive")
    if args.max_mosaics is not None and args.max_mosaics < 1:
        parser.error("--max-mosaics must be positive")
    return args


def main() -> int:
    args = parse_args()
    if not args.input_root.is_dir():
        raise NotADirectoryError(args.input_root)
    jobs = discover_jobs(args.input_root, args.patients)
    if args.visits:
        selected_visits = set(args.visits)
        jobs = [job for job in jobs if job.visit in selected_visits]
    if args.mosaics:
        selected_mosaics = set(args.mosaics)
        jobs = [job for job in jobs if job.mosaic in selected_mosaics]
    if args.max_mosaics is not None:
        jobs = jobs[: args.max_mosaics]
    if not jobs:
        raise RuntimeError("No final calibrated phasor TIFFs were discovered")

    args.output_root.mkdir(parents=True, exist_ok=True)
    normalization_suffix = (
        "_mosaic_normalized"
        if args.normalization_mode == "mosaic"
        else "_tile_normalized"
    )
    pdf_path = args.output_root / (
        f"all_patients_visits_phase_rgb_mosaics{normalization_suffix}.pdf"
    )
    partial_pdf = pdf_path.with_name(f".{pdf_path.name}.partial.pdf")
    manifest_path = args.output_root / "phase_rgb_manifest.csv"
    rows: list[dict[str, Any]] = []
    print(f"Mosaics: {len(jobs)}")
    print(f"Input:   {args.input_root}")
    print(f"Output:  {args.output_root}")
    print(
        f"Phase: {args.phase_min_deg:g}-{args.phase_max_deg:g} degrees; "
        f"green={GREEN_SCALE}; blue={BLUE_SCALE}; composite=DC weighted; "
        f"normalization={args.normalization_mode}"
    )

    partial_pdf.unlink(missing_ok=True)
    try:
        with PdfPages(partial_pdf) as pdf:
            cover = make_cover(jobs, args)
            pdf.savefig(cover, dpi=args.pdf_dpi, facecolor="white")
            plt.close(cover)
            for index, job in enumerate(jobs, start=1):
                print(
                    f"[{index}/{len(jobs)}] {job.patient} | {job.visit} | "
                    f"{job.mosaic}",
                    flush=True,
                )
                green_rgb, blue_rgb, composite_rgb, dimensions = build_rgb_mosaics(
                    job, args
                )
                paths = output_paths(args.output_root, job)
                save_rgb_png(paths.green, green_rgb, args.png_compress_level)
                save_rgb_png(paths.blue, blue_rgb, args.png_compress_level)
                save_rgb_png(paths.composite, composite_rgb, args.png_compress_level)
                figure = make_triptych(job, green_rgb, blue_rgb, composite_rgb, args)
                paths.triptych.parent.mkdir(parents=True, exist_ok=True)
                figure.savefig(
                    paths.triptych,
                    dpi=args.triptych_dpi,
                    bbox_inches="tight",
                    facecolor="white",
                )
                pdf.savefig(figure, dpi=args.pdf_dpi, facecolor="white")
                plt.close(figure)
                rows.append(
                    {
                        "patient": job.patient,
                        "visit": job.visit,
                        "mosaic": job.mosaic,
                        "acquisition_type": job.acquisition_type,
                        "channels": "+".join(job.channels),
                        "tile_count": len(job.tile_numbers),
                        **dimensions,
                        "phase_min_deg": args.phase_min_deg,
                        "phase_max_deg": args.phase_max_deg,
                        "phase_gamma": args.phase_gamma,
                        "normalization_mode": args.normalization_mode,
                        "intensity_pmin": args.intensity_pmin,
                        "intensity_pmax": args.intensity_pmax,
                        "intensity_gamma": args.intensity_gamma,
                        "tile_reference_percentile": (args.tile_reference_percentile),
                        "tile_gain_minimum": args.tile_gain_minimum,
                        "tile_gain_maximum": args.tile_gain_maximum,
                        "green_png": str(paths.green),
                        "blue_png": str(paths.blue),
                        "composite_png": str(paths.composite),
                        "triptych_png": str(paths.triptych),
                        "source_tiff": str(job.tiff_path),
                    }
                )
                write_manifest(manifest_path, rows)
                del green_rgb, blue_rgb, composite_rgb
    except BaseException:
        partial_pdf.unlink(missing_ok=True)
        raise

    pdf_path.unlink(missing_ok=True)
    partial_pdf.replace(pdf_path)
    print(f"PNGs:     {args.output_root}")
    print(f"PDF:      {pdf_path}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
