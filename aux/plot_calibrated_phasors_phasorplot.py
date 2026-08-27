#!/usr/bin/env python3
"""Plot final calibrated FLIM phasors with PhasorPlot.hist2d.

This independent visualization step reads the final calibrated, filtered,
non-thresholded TIFFs, retains the brightest DC fraction in each tile for
plotting only, writes one PNG per mosaic, and combines all mosaics into a PDF.
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
import matplotlib.patheffects as path_effects  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tifffile  # noqa: E402
from matplotlib import colormaps  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from phasorpy.plot import PhasorPlot  # noqa: E402

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from calibration_by_blue.calibrate_phasors import (  # noqa: E402
    CHANNEL_COLORS,
    COMPONENTS,
    LASER_FREQUENCY_MHZ,
    LIFETIME_RANGES_NS,
    brightest_dc_mask,
    theoretical_segment,
)
from utils.color_scales import get_phase_colormap  # noqa: E402

DEFAULT_INPUT_ROOT = Path(
    "/Users/schutyb/Documents/balu_lab/dod/data_curated/calibrated_filtered_phasor"
)
DEFAULT_PATIENTS = ("p427", "p437", "p439", "p449", "p476")
METADATA_SUFFIX = "_calibrated_filtered_phasor.json"
TIFF_SUFFIX = "_calibrated_filtered_phasor.tiff"
MANIFEST_FIELDS = (
    "patient",
    "visit",
    "mosaic",
    "acquisition_type",
    "channel",
    "tiles",
    "valid_pixels",
    "selected_pixels",
    "selected_percent_of_valid",
    "histogram_min_count",
    "minimum_tile_dc_threshold",
    "median_tile_dc_threshold",
    "maximum_tile_dc_threshold",
    "source_tiff",
    "output_png",
)


@dataclass(frozen=True)
class PlotJob:
    patient: str
    visit: str
    mosaic: str
    acquisition_type: str
    channels: tuple[str, ...]
    tile_numbers: tuple[int, ...]
    tiff_path: Path


def natural_key(value: str | Path) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", str(value))
    )


def discover_jobs(input_root: Path, patients: list[str]) -> list[PlotJob]:
    jobs: list[PlotJob] = []
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
            tiff_path = metadata_path.with_name(
                metadata_path.name.removesuffix(METADATA_SUFFIX) + TIFF_SUFFIX
            )
            if not tiff_path.is_file():
                raise FileNotFoundError(tiff_path)
            if not channels or any(
                value not in {"green", "blue"} for value in channels
            ):
                raise ValueError(f"Unexpected channels in {metadata_path}: {channels}")
            jobs.append(
                PlotJob(
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


def output_png_path(
    output_root: Path, job: PlotJob, show_calibration_lines: bool
) -> Path:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", job.mosaic).strip("._")
    suffix = "_calibration_lines" if show_calibration_lines else ""
    return (
        output_root / job.patient / job.visit / f"{stem}_phasorplot_hist2d{suffix}.png"
    )


def resolve_colormap(name: str) -> Any:
    """Return a project FLIM colormap or a standard Matplotlib colormap."""
    try:
        return get_phase_colormap(name, n=2048)
    except ValueError:
        try:
            return colormaps[name]
        except KeyError as error:
            raise ValueError(f"Unknown colormap: {name}") from error


def collect_channel_points(
    pages: Any,
    job: PlotJob,
    channel_index: int,
    top_dc_percent: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    real_parts: list[np.ndarray] = []
    imag_parts: list[np.ndarray] = []
    thresholds: list[float] = []
    total_valid = 0
    total_selected = 0
    tile_count = len(job.tile_numbers)
    for tile_index in range(tile_count):
        page = (channel_index * tile_count + tile_index) * len(COMPONENTS)
        mean = np.asarray(pages[page].asarray(), dtype=np.float32)
        real = np.asarray(pages[page + 1].asarray(), dtype=np.float32)
        imag = np.asarray(pages[page + 2].asarray(), dtype=np.float32)
        mask, threshold, valid_count = brightest_dc_mask(
            mean, real, imag, top_dc_percent
        )
        real_parts.append(np.asarray(real[mask], dtype=np.float32))
        imag_parts.append(np.asarray(imag[mask], dtype=np.float32))
        thresholds.append(threshold)
        total_valid += valid_count
        total_selected += int(np.count_nonzero(mask))
    return (
        np.concatenate(real_parts),
        np.concatenate(imag_parts),
        {
            "tiles": tile_count,
            "valid_pixels": total_valid,
            "selected_pixels": total_selected,
            "selected_percent_of_valid": 100.0 * total_selected / total_valid,
            "minimum_tile_dc_threshold": min(thresholds),
            "median_tile_dc_threshold": float(np.median(thresholds)),
            "maximum_tile_dc_threshold": max(thresholds),
        },
    )


def add_calibration_line(axis: plt.Axes, channel: str) -> None:
    """Overlay the lifetime segment used to calibrate the given channel."""
    start, end = theoretical_segment(channel)
    lifetime_start, lifetime_end = LIFETIME_RANGES_NS[channel]
    color = CHANNEL_COLORS[channel]
    line = axis.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color=color,
        linestyle="--",
        linewidth=2.4,
        marker="o",
        markersize=5.5,
        markerfacecolor="white",
        markeredgewidth=1.6,
        zorder=20,
        label=(
            f"{channel.capitalize()} calibration: "
            f"{lifetime_start:g}-{lifetime_end:g} ns"
        ),
    )[0]
    line.set_path_effects(
        [path_effects.Stroke(linewidth=4.2, foreground="white"), path_effects.Normal()]
    )
    for (g_value, s_value), lifetime, offset in (
        (start, lifetime_start, (-8, 9)),
        (end, lifetime_end, (7, -13)),
    ):
        annotation = axis.annotate(
            f"{lifetime:g} ns",
            (g_value, s_value),
            xytext=offset,
            textcoords="offset points",
            color=color,
            fontsize=8.5,
            fontweight="bold",
            ha="right" if offset[0] < 0 else "left",
            va="bottom" if offset[1] > 0 else "top",
            zorder=21,
        )
        annotation.set_path_effects(
            [
                path_effects.Stroke(linewidth=2.8, foreground="white"),
                path_effects.Normal(),
            ]
        )
    axis.legend(
        loc="lower left",
        fontsize=8.5,
        frameon=True,
        facecolor="white",
        framealpha=0.92,
        edgecolor="#777777",
    )


def plot_job(
    job: PlotJob,
    output_png: Path,
    top_dc_percent: float,
    bins: int,
    histogram_min_count: int,
    dpi: int,
    colormap_name: str,
    show_calibration_lines: bool,
) -> tuple[plt.Figure, list[dict[str, Any]]]:
    channel_count = len(job.channels)
    figure, axes = plt.subplots(
        1,
        channel_count,
        figsize=(7.3 * channel_count, 6.7),
        squeeze=False,
        constrained_layout=True,
    )
    colormap = resolve_colormap(colormap_name)
    rows: list[dict[str, Any]] = []
    with tifffile.TiffFile(job.tiff_path) as tif:
        series = tif.series[0]
        if np.dtype(series.dtype) != np.dtype(np.float32):
            raise ValueError(f"Expected float32, found {series.dtype}: {job.tiff_path}")
        pages = series.pages
        expected_pages = len(job.channels) * len(job.tile_numbers) * len(COMPONENTS)
        if len(pages) != expected_pages:
            raise ValueError(
                f"Expected {expected_pages} pages, found {len(pages)}: {job.tiff_path}"
            )
        for channel_index, channel in enumerate(job.channels):
            axis = axes[0, channel_index]
            phasor_plot = PhasorPlot(
                ax=axis,
                frequency=LASER_FREQUENCY_MHZ,
                grid={"color": "#606060", "linewidth": 0.8, "alpha": 0.75},
            )
            real, imag, statistics = collect_channel_points(
                pages, job, channel_index, top_dc_percent
            )
            phasor_plot.hist2d(
                real,
                imag,
                bins=bins,
                cmap=colormap,
                norm="log",
                cmin=histogram_min_count,
                rasterized=True,
                shading="auto",
            )
            mesh = axis.collections[-1]
            colorbar = figure.colorbar(mesh, ax=axis, fraction=0.046, pad=0.04)
            colorbar.set_label("Pixel count per bin (log scale)")
            if show_calibration_lines:
                add_calibration_line(axis, channel)
            axis.set_title(
                f"{channel.capitalize()} | {len(job.tile_numbers)} tiles\n"
                f"brightest {top_dc_percent:g}% DC per tile | "
                f"hist bins ≥ {histogram_min_count} counts | "
                f"n DC={statistics['selected_pixels']:,}",
                fontsize=11,
            )
            rows.append(
                {
                    "patient": job.patient,
                    "visit": job.visit,
                    "mosaic": job.mosaic,
                    "acquisition_type": job.acquisition_type,
                    "channel": channel,
                    **statistics,
                    "histogram_min_count": histogram_min_count,
                    "source_tiff": str(job.tiff_path),
                    "output_png": str(output_png),
                }
            )
            del real, imag
    figure.suptitle(
        f"{job.patient} | {job.visit} | {job.mosaic} | {job.acquisition_type}\n"
        "Calibrated + median 7x7 twice | hist2d visualization only"
        + (" | calibration lines" if show_calibration_lines else ""),
        fontsize=13,
        fontweight="bold",
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    return figure, rows


def cover_page(
    jobs: list[PlotJob],
    top_dc_percent: float,
    histogram_min_count: int,
    colormap_name: str,
    show_calibration_lines: bool,
) -> plt.Figure:
    patients = sorted({job.patient for job in jobs}, key=natural_key)
    visits = sorted({(job.patient, job.visit) for job in jobs}, key=natural_key)
    figure = plt.figure(figsize=(11.69, 8.27), constrained_layout=True)
    axis = figure.subplots()
    axis.axis("off")
    axis.text(
        0.5,
        0.73,
        (
            "Calibrated FLIM phasors with calibration lines"
            if show_calibration_lines
            else "Calibrated FLIM phasors"
        ),
        ha="center",
        va="center",
        fontsize=28,
        fontweight="bold",
    )
    axis.text(
        0.5,
        0.57,
        "PhasorPy PhasorPlot.hist2d",
        ha="center",
        va="center",
        fontsize=19,
    )
    axis.text(
        0.5,
        0.39,
        (
            f"Patients: {', '.join(patients)}\n"
            f"Patient-visits: {len(visits)} | Mosaics: {len(jobs)}\n"
            f"Brightest {top_dc_percent:g}% DC selected independently per tile\n"
            f"Histogram bins with fewer than {histogram_min_count} counts hidden\n"
            f"Density colormap: {colormap_name}"
            + (
                "\nBlue calibration line: 0-3.5 ns | "
                "Green calibration line: 3.5-0.1 ns"
                if show_calibration_lines
                else ""
            )
        ),
        ha="center",
        va="center",
        fontsize=15,
        linespacing=1.6,
    )
    axis.text(
        0.5,
        0.16,
        "Thresholding affects these plots only; source TIFFs remain unchanged.",
        ha="center",
        va="center",
        fontsize=11,
        color="#444444",
    )
    return figure


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--patients", nargs="+", default=list(DEFAULT_PATIENTS))
    parser.add_argument("--visits", nargs="+")
    parser.add_argument("--mosaics", nargs="+")
    parser.add_argument("--top-dc-percent", type=float, default=35.0)
    parser.add_argument("--bins", type=int, default=256)
    parser.add_argument("--histogram-min-count", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--pdf-dpi", type=int, default=180)
    parser.add_argument("--colormap", default="RdYlGn_r")
    parser.add_argument(
        "--show-calibration-lines",
        action="store_true",
        help="Overlay the lifetime segment used to calibrate each channel.",
    )
    parser.add_argument("--max-mosaics", type=int)
    args = parser.parse_args()
    args.input_root = args.input_root.expanduser().resolve()
    args.output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else args.input_root / "phasorplot_hist2d_top35"
    )
    if not 0 < args.top_dc_percent <= 100:
        parser.error("--top-dc-percent must be in (0, 100]")
    if args.bins < 2 or args.dpi < 1 or args.pdf_dpi < 1:
        parser.error("bins and DPI values must be positive")
    if args.histogram_min_count < 1:
        parser.error("--histogram-min-count must be positive")
    if args.max_mosaics is not None and args.max_mosaics < 1:
        parser.error("--max-mosaics must be positive")
    resolve_colormap(args.colormap)
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
    line_suffix = "_calibration_lines" if args.show_calibration_lines else ""
    pdf_path = args.output_root / (
        "all_patients_visits_phasorplot_hist2d"
        f"_cmin{args.histogram_min_count}{line_suffix}.pdf"
    )
    partial_pdf = pdf_path.with_name(f".{pdf_path.name}.partial.pdf")
    manifest_path = args.output_root / "phasorplot_hist2d_manifest.csv"
    rows: list[dict[str, Any]] = []
    print(f"Mosaics: {len(jobs)}")
    print(f"Input:   {args.input_root}")
    print(f"Output:  {args.output_root}")
    print(
        f"Plot: PhasorPlot.hist2d, brightest {args.top_dc_percent:g}% DC per tile, "
        f"histogram cmin={args.histogram_min_count}, cmap={args.colormap}"
        + (", calibration lines=on" if args.show_calibration_lines else "")
    )

    partial_pdf.unlink(missing_ok=True)
    try:
        with PdfPages(partial_pdf) as pdf:
            cover = cover_page(
                jobs,
                args.top_dc_percent,
                args.histogram_min_count,
                args.colormap,
                args.show_calibration_lines,
            )
            pdf.savefig(cover, dpi=args.pdf_dpi, facecolor="white")
            plt.close(cover)
            for index, job in enumerate(jobs, start=1):
                print(
                    f"[{index}/{len(jobs)}] {job.patient} | {job.visit} | {job.mosaic}",
                    flush=True,
                )
                png_path = output_png_path(
                    args.output_root, job, args.show_calibration_lines
                )
                figure, job_rows = plot_job(
                    job,
                    png_path,
                    args.top_dc_percent,
                    args.bins,
                    args.histogram_min_count,
                    args.dpi,
                    args.colormap,
                    args.show_calibration_lines,
                )
                pdf.savefig(figure, dpi=args.pdf_dpi, facecolor="white")
                plt.close(figure)
                rows.extend(job_rows)
                write_manifest(manifest_path, rows)
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
