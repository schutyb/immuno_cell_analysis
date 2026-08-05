#!/usr/bin/env python3
"""Analyze Coumarin references across patients, visits, and modalities.

The raw-data path discovers visit folders named like ``p427-v01``. Inside
each experiment it recognizes simultaneous ``*32Sp`` acquisitions and
sequential ``*32A1`` (green) / ``*32A0`` (blue) acquisitions.

For every channel acquisition the established Coumarin workflow is applied:

* correct detector bins;
* cubic spatial downsample/upsample, removing 50 pixels by default;
* calculate first-harmonic DC/G/S;
* apply a 5x5 median filter twice;
* remove the lowest 25% of finite positive DC;
* calculate the mean phasor center and the phase correction required to place
  Coumarin at its theoretical 2.5 ns position at 80 MHz.

Paired green/blue corrections are then used to calculate both the signed
``green - blue`` difference and its positive magnitude. Summary CSVs report
means by modality and by patient/visit/modality. Bar plots retain the signed
difference, matching the historical analysis, while annotating the positive
offset used by the production calibration.

Use ``--channel-csv`` to regenerate differences, summaries, and plots from a
previous channel-calibration table without reading the raw TIFFs again.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import pandas as pd
from phasorpy.filter import phasor_filter_median
from phasorpy.lifetime import phasor_from_lifetime
from phasorpy.phasor import phasor_center, phasor_from_signal

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from calibration_by_blue.flim_io import (  # noqa: E402
    extract_tile_number,
    natural_key,
    normalize_single_detector_32_bins,
    prepare_tile,
    split_green_blue,
)
from calibration_by_blue.flim_preprocessing import (  # noqa: E402
    cubic_downsample_upsample,
)

DEFAULT_DATA_ROOT = Path("/Users/schutyb/Documents/balu_lab/dod/data_curated")
FREQUENCY_MHZ = 80.0
COUMARIN_LIFETIME_NS = 2.5
DEFAULT_DOWNSAMPLE_PIXELS = 50
DEFAULT_FILTER_SIZE = 5
DEFAULT_FILTER_REPEAT = 2
DEFAULT_DC_PERCENTILE_REMOVED = 25.0

CHANNEL_FIELDS = (
    "patient",
    "visit",
    "visit_folder",
    "experiment",
    "job_id",
    "acquisition_type",
    "pair_index",
    "replicate_index",
    "channel",
    "source_tiff",
    "frequency_mhz",
    "coumarin_lifetime_ns",
    "spatial_height",
    "spatial_width",
    "lifetime_bins",
    "downsample_pixels",
    "filter_size",
    "filter_repeat",
    "dc_percentile_removed",
    "dc_cutoff",
    "number_of_reference_pixels",
    "measured_center_g",
    "measured_center_s",
    "theoretical_g",
    "theoretical_s",
    "measured_phase_deg",
    "expected_phase_deg",
    "phase_correction_deg",
    "measured_radius",
    "expected_radius",
    "modulation_scale",
)

DIFFERENCE_FIELDS = (
    "patient",
    "visit",
    "visit_folder",
    "experiment",
    "job_id",
    "acquisition_type",
    "pair_index",
    "replicate_index",
    "green_phase_correction_deg",
    "blue_phase_correction_deg",
    "green_minus_blue_correction_deg",
    "positive_offset_magnitude_deg",
)


@dataclass(frozen=True)
class CoumarinJob:
    patient: str
    visit: str
    visit_folder: str
    experiment: str
    acquisition_type: str
    pair_index: int
    replicate_index: int
    green_tiff: Path | None
    blue_tiff: Path | None

    @property
    def job_id(self) -> str:
        parts = (
            self.patient,
            self.visit,
            self.experiment,
            self.acquisition_type,
            f"pair{self.pair_index:02d}",
            f"rep{self.replicate_index:02d}",
        )
        return "__".join(sanitize(value) for value in parts)


def sanitize(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "unknown"


def parse_visit_folder(name: str) -> tuple[str, str] | None:
    match = re.fullmatch(r"(p\d+)-v(\d+)", name, flags=re.IGNORECASE)
    if match is None:
        return None
    return match.group(1).lower(), f"visit{int(match.group(2)):02d}"


def acquisition_kind(path: Path) -> str | None:
    name = path.name
    if re.search(r"32Sp$", name, flags=re.IGNORECASE):
        return "Sp"
    if re.search(r"32A1$", name, flags=re.IGNORECASE):
        return "A1"
    if re.search(r"32A0$", name, flags=re.IGNORECASE):
        return "A0"
    return None


def acquisition_signature(path: Path) -> str:
    name = re.sub(r"^Image\d+_", "", path.name, flags=re.IGNORECASE)
    name = re.sub(r"_?32A[01]$", "", name, flags=re.IGNORECASE)
    return name.casefold()


def acquisition_tiffs(directory: Path) -> list[Path]:
    paths = [
        path
        for path in directory.iterdir()
        if path.is_file() and extract_tile_number(path) is not None
    ]
    return sorted(paths, key=natural_key)


def discover_jobs(coumarin_root: Path) -> tuple[list[CoumarinJob], list[str]]:
    jobs: list[CoumarinJob] = []
    warnings: list[str] = []
    for visit_dir in sorted(
        (path for path in coumarin_root.iterdir() if path.is_dir()),
        key=natural_key,
    ):
        parsed = parse_visit_folder(visit_dir.name)
        if parsed is None:
            warnings.append(f"Ignored non-visit folder: {visit_dir}")
            continue
        patient, visit = parsed
        for experiment_dir in sorted(
            (path for path in visit_dir.iterdir() if path.is_dir()),
            key=natural_key,
        ):
            acquisition_dirs = [
                path
                for path in experiment_dir.iterdir()
                if path.is_dir() and acquisition_kind(path) is not None
            ]

            split_dirs = sorted(
                (path for path in acquisition_dirs if acquisition_kind(path) == "Sp"),
                key=natural_key,
            )
            for pair_index, directory in enumerate(split_dirs, start=1):
                tiffs = acquisition_tiffs(directory)
                if not tiffs:
                    warnings.append(
                        f"No TIFFs in simultaneous acquisition: {directory}"
                    )
                for replicate_index, source in enumerate(tiffs, start=1):
                    jobs.append(
                        CoumarinJob(
                            patient=patient,
                            visit=visit,
                            visit_folder=visit_dir.name,
                            experiment=experiment_dir.name,
                            acquisition_type="Sp",
                            pair_index=pair_index,
                            replicate_index=replicate_index,
                            green_tiff=source,
                            blue_tiff=source,
                        )
                    )

            grouped: dict[str, dict[str, list[Path]]] = {}
            for directory in acquisition_dirs:
                kind = acquisition_kind(directory)
                if kind not in {"A1", "A0"}:
                    continue
                signature = acquisition_signature(directory)
                grouped.setdefault(signature, {"A1": [], "A0": []})[kind].append(
                    directory
                )

            pair_index = 0
            for signature, channels in sorted(grouped.items()):
                green_dirs = sorted(channels["A1"], key=natural_key)
                blue_dirs = sorted(channels["A0"], key=natural_key)
                directory_count = max(len(green_dirs), len(blue_dirs))
                for directory_index in range(directory_count):
                    pair_index += 1
                    green_dir = (
                        green_dirs[directory_index]
                        if directory_index < len(green_dirs)
                        else None
                    )
                    blue_dir = (
                        blue_dirs[directory_index]
                        if directory_index < len(blue_dirs)
                        else None
                    )
                    green_tiffs = acquisition_tiffs(green_dir) if green_dir else []
                    blue_tiffs = acquisition_tiffs(blue_dir) if blue_dir else []
                    replicate_count = max(len(green_tiffs), len(blue_tiffs))
                    if replicate_count == 0:
                        warnings.append(
                            f"No TIFFs for A1/A0 signature {signature}: "
                            f"{experiment_dir}"
                        )
                    for replicate_index in range(replicate_count):
                        jobs.append(
                            CoumarinJob(
                                patient=patient,
                                visit=visit,
                                visit_folder=visit_dir.name,
                                experiment=experiment_dir.name,
                                acquisition_type="A1_A0",
                                pair_index=pair_index,
                                replicate_index=replicate_index + 1,
                                green_tiff=(
                                    green_tiffs[replicate_index]
                                    if replicate_index < len(green_tiffs)
                                    else None
                                ),
                                blue_tiff=(
                                    blue_tiffs[replicate_index]
                                    if replicate_index < len(blue_tiffs)
                                    else None
                                ),
                            )
                        )
    jobs.sort(
        key=lambda job: natural_key(
            f"{job.patient}/{job.visit}/{job.experiment}/"
            f"{job.acquisition_type}/{job.pair_index}/{job.replicate_index}"
        )
    )
    return jobs, warnings


def corrected_decays(job: CoumarinJob) -> dict[str, tuple[np.ndarray, Path]]:
    if job.acquisition_type == "Sp":
        if job.green_tiff is None:
            raise ValueError("Sp job has no source TIFF")
        raw, _, _ = prepare_tile(job.green_tiff)
        green, blue = split_green_blue(raw)
        return {
            "green": (green, job.green_tiff),
            "blue": (blue, job.green_tiff),
        }
    result: dict[str, tuple[np.ndarray, Path]] = {}
    if job.green_tiff is not None:
        raw, _, _ = prepare_tile(job.green_tiff)
        result["green"] = (normalize_single_detector_32_bins(raw), job.green_tiff)
    if job.blue_tiff is not None:
        raw, _, _ = prepare_tile(job.blue_tiff)
        result["blue"] = (normalize_single_detector_32_bins(raw), job.blue_tiff)
    return result


def analyze_channel(
    decay: np.ndarray,
    source: Path,
    job: CoumarinJob,
    channel: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    decay = cubic_downsample_upsample(
        decay,
        pixels_to_remove=args.downsample_pixels,
        workers=args.resample_workers,
    )
    mean, real, imag = phasor_from_signal(
        np.asarray(decay, dtype=np.float32),
        axis=-1,
        harmonic=1,
        use_fft=False,
        dtype=np.float32,
        normalize=True,
        num_threads=0,
    )
    mean_f, real_f, imag_f = phasor_filter_median(
        mean,
        real,
        imag,
        size=args.filter_size,
        repeat=args.filter_repeat,
        use_scipy=False,
        num_threads=0,
    )
    valid = (
        np.isfinite(mean_f) & np.isfinite(real_f) & np.isfinite(imag_f) & (mean_f > 0)
    )
    if not np.any(valid):
        raise ValueError("No finite positive-DC Coumarin pixels")
    cutoff = float(np.percentile(mean_f[valid], args.dc_percentile_removed))
    selected = valid & (mean_f > cutoff)
    if not np.any(selected):
        raise ValueError("DC threshold removed every Coumarin pixel")
    mean_masked = np.where(selected, mean_f, np.nan)
    real_masked = np.where(selected, real_f, np.nan)
    imag_masked = np.where(selected, imag_f, np.nan)
    _, center_g, center_s = phasor_center(
        mean_masked,
        real_masked,
        imag_masked,
        method="mean",
        nan_safe=True,
    )
    center_g, center_s = float(center_g), float(center_s)
    theoretical_g, theoretical_s = phasor_from_lifetime(
        args.frequency_mhz, args.coumarin_lifetime_ns
    )
    theoretical_g, theoretical_s = float(theoretical_g), float(theoretical_s)
    measured_phase = math.degrees(math.atan2(center_s, center_g))
    expected_phase = math.degrees(math.atan2(theoretical_s, theoretical_g))
    correction = math.degrees(
        math.atan2(
            math.sin(math.radians(expected_phase - measured_phase)),
            math.cos(math.radians(expected_phase - measured_phase)),
        )
    )
    measured_radius = math.hypot(center_g, center_s)
    expected_radius = math.hypot(theoretical_g, theoretical_s)
    return {
        "patient": job.patient,
        "visit": job.visit,
        "visit_folder": job.visit_folder,
        "experiment": job.experiment,
        "job_id": job.job_id,
        "acquisition_type": job.acquisition_type,
        "pair_index": job.pair_index,
        "replicate_index": job.replicate_index,
        "channel": channel,
        "source_tiff": str(source),
        "frequency_mhz": args.frequency_mhz,
        "coumarin_lifetime_ns": args.coumarin_lifetime_ns,
        "spatial_height": decay.shape[0],
        "spatial_width": decay.shape[1],
        "lifetime_bins": decay.shape[-1],
        "downsample_pixels": args.downsample_pixels,
        "filter_size": args.filter_size,
        "filter_repeat": args.filter_repeat,
        "dc_percentile_removed": args.dc_percentile_removed,
        "dc_cutoff": cutoff,
        "number_of_reference_pixels": int(np.count_nonzero(selected)),
        "measured_center_g": center_g,
        "measured_center_s": center_s,
        "theoretical_g": theoretical_g,
        "theoretical_s": theoretical_s,
        "measured_phase_deg": measured_phase,
        "expected_phase_deg": expected_phase,
        "phase_correction_deg": correction,
        "measured_radius": measured_radius,
        "expected_radius": expected_radius,
        "modulation_scale": expected_radius / measured_radius,
    }


def differences_from_channels(channels: pd.DataFrame) -> pd.DataFrame:
    required = {
        "patient",
        "visit",
        "visit_folder",
        "experiment",
        "job_id",
        "acquisition_type",
        "pair_index",
        "replicate_index",
        "channel",
        "phase_correction_deg",
    }
    missing = required.difference(channels.columns)
    if missing:
        raise ValueError(f"Channel table is missing: {', '.join(sorted(missing))}")
    index_columns = [
        "patient",
        "visit",
        "visit_folder",
        "experiment",
        "job_id",
        "acquisition_type",
        "pair_index",
        "replicate_index",
    ]
    pivot = channels.pivot_table(
        index=index_columns,
        columns="channel",
        values="phase_correction_deg",
        aggfunc="first",
    ).reset_index()
    if "green" not in pivot or "blue" not in pivot:
        return pd.DataFrame(columns=DIFFERENCE_FIELDS)
    paired = pivot.dropna(subset=["green", "blue"]).copy()
    paired = paired.rename(
        columns={
            "green": "green_phase_correction_deg",
            "blue": "blue_phase_correction_deg",
        }
    )
    paired["green_minus_blue_correction_deg"] = (
        paired["green_phase_correction_deg"] - paired["blue_phase_correction_deg"]
    )
    paired["positive_offset_magnitude_deg"] = paired[
        "green_minus_blue_correction_deg"
    ].abs()
    return paired[list(DIFFERENCE_FIELDS)].sort_values(
        ["acquisition_type", "patient", "visit", "experiment", "pair_index"]
    )


def modality_summary(channels: pd.DataFrame, differences: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    modalities = sorted(
        set(channels["acquisition_type"].dropna())
        | set(differences["acquisition_type"].dropna()),
        key=natural_key,
    )
    for modality in modalities:
        channel_group = channels[channels["acquisition_type"] == modality]
        difference_group = differences[differences["acquisition_type"] == modality]
        green = channel_group.loc[
            channel_group["channel"] == "green", "phase_correction_deg"
        ].astype(float)
        blue = channel_group.loc[
            channel_group["channel"] == "blue", "phase_correction_deg"
        ].astype(float)
        signed = difference_group["green_minus_blue_correction_deg"].astype(float)
        rows.append(
            {
                "acquisition_type": modality,
                "number_of_green": len(green),
                "number_of_blue": len(blue),
                "number_of_pairs": len(signed),
                "mean_green_correction_deg": green.mean(),
                "mean_blue_correction_deg": blue.mean(),
                "mean_green_minus_blue_deg": signed.mean(),
                "mean_positive_offset_deg": signed.abs().mean(),
                "recommended_green_offset_deg": abs(signed.mean()),
                "median_green_minus_blue_deg": signed.median(),
                "standard_deviation_deg": signed.std(ddof=1),
            }
        )
    return pd.DataFrame(rows)


def patient_visit_summary(differences: pd.DataFrame) -> pd.DataFrame:
    if differences.empty:
        return pd.DataFrame()
    grouped = differences.groupby(["patient", "visit", "acquisition_type"], sort=True)
    return grouped.agg(
        number_of_pairs=("green_minus_blue_correction_deg", "size"),
        mean_green_correction_deg=("green_phase_correction_deg", "mean"),
        mean_blue_correction_deg=("blue_phase_correction_deg", "mean"),
        mean_green_minus_blue_deg=("green_minus_blue_correction_deg", "mean"),
        mean_positive_offset_deg=("positive_offset_magnitude_deg", "mean"),
        median_green_minus_blue_deg=("green_minus_blue_correction_deg", "median"),
        standard_deviation_deg=("green_minus_blue_correction_deg", "std"),
    ).reset_index()


def plot_differences(
    differences: pd.DataFrame,
    summary: pd.DataFrame,
    output: Path,
    dpi: int,
) -> None:
    modalities = list(summary["acquisition_type"])
    if not modalities:
        raise ValueError("No paired green/blue differences are available")
    figure, axes = plt.subplots(
        len(modalities),
        1,
        figsize=(max(12, len(differences) * 0.45), 5.2 * len(modalities)),
        squeeze=False,
    )
    colors = {"Sp": "#3b82f6", "A1_A0": "#f59e0b"}
    for axis, modality in zip(axes[:, 0], modalities, strict=True):
        group = differences[differences["acquisition_type"] == modality].copy()
        group = group.sort_values(["patient", "visit", "experiment", "pair_index"])
        values = group["green_minus_blue_correction_deg"].to_numpy(float)
        labels = [
            f"{row.patient} {row.visit}\n{row.experiment} #{int(row.pair_index)}"
            for row in group.itertuples()
        ]
        positions = np.arange(len(group))
        axis.bar(
            positions,
            values,
            color=colors.get(modality, "#64748b"),
            edgecolor="black",
            linewidth=0.4,
        )
        mean_signed = float(values.mean())
        axis.axhline(
            mean_signed,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=(
                f"mean green−blue = {mean_signed:+.3f}°; "
                f"positive magnitude = {abs(mean_signed):.3f}°"
            ),
        )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(positions, labels, rotation=55, ha="right", fontsize=8)
        axis.set_ylabel("Green correction − blue correction (degrees)")
        axis.set_title(f"Coumarin correction difference | {modality}")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def write_csv(
    path: Path, rows: Iterable[dict[str, Any]], fields: Iterable[str]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve_coumarin_root(data_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    candidates = (
        data_root / "coumarin",
        data_root / "self_calibration_coumarin_by_visit" / "coumarin",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--coumarin-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--channel-csv",
        type=Path,
        help="Skip raw processing and summarize an existing channel CSV.",
    )
    parser.add_argument("--patients", nargs="*")
    parser.add_argument("--frequency-mhz", type=float, default=FREQUENCY_MHZ)
    parser.add_argument(
        "--coumarin-lifetime-ns", type=float, default=COUMARIN_LIFETIME_NS
    )
    parser.add_argument(
        "--downsample-pixels", type=int, default=DEFAULT_DOWNSAMPLE_PIXELS
    )
    parser.add_argument("--resample-workers", type=int, default=4)
    parser.add_argument("--filter-size", type=int, default=DEFAULT_FILTER_SIZE)
    parser.add_argument("--filter-repeat", type=int, default=DEFAULT_FILTER_REPEAT)
    parser.add_argument(
        "--dc-percentile-removed",
        type=float,
        default=DEFAULT_DC_PERCENTILE_REMOVED,
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.coumarin_root = resolve_coumarin_root(args.data_root, args.coumarin_root)
    args.output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else args.data_root / "coumarin_analysis"
    )
    if args.channel_csv:
        args.channel_csv = args.channel_csv.expanduser().resolve()
    if args.frequency_mhz <= 0 or args.coumarin_lifetime_ns <= 0:
        parser.error("frequency and lifetime must be positive")
    if args.downsample_pixels < 1 or args.resample_workers < 1:
        parser.error("resampling values must be positive")
    if args.filter_size < 1 or args.filter_size % 2 == 0:
        parser.error("--filter-size must be a positive odd integer")
    if args.filter_repeat < 1:
        parser.error("--filter-repeat must be positive")
    if not 0 <= args.dc_percentile_removed < 100:
        parser.error("--dc-percentile-removed must be in [0, 100)")
    return args


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    errors: list[dict[str, str]] = []
    channel_csv = args.output_root / "coumarin_channel_calibration.csv"
    if args.channel_csv is not None:
        if not args.channel_csv.is_file():
            raise FileNotFoundError(args.channel_csv)
        channels = pd.read_csv(args.channel_csv)
        # Accept the historical column name without manual editing.
        if "phase_shift_deg" in channels and "phase_correction_deg" not in channels:
            channels = channels.rename(
                columns={"phase_shift_deg": "phase_correction_deg"}
            )
        channels.to_csv(channel_csv, index=False)
    else:
        if not args.coumarin_root.is_dir():
            raise NotADirectoryError(args.coumarin_root)
        jobs, warnings = discover_jobs(args.coumarin_root)
        for warning in warnings:
            print(f"[WARN] {warning}")
        if args.patients:
            selected = {patient.casefold() for patient in args.patients}
            jobs = [job for job in jobs if job.patient.casefold() in selected]
        if not jobs:
            raise RuntimeError("No Coumarin Sp or A1/A0 acquisitions were found")
        rows: list[dict[str, Any]] = []
        for index, job in enumerate(jobs, start=1):
            print(
                f"[{index}/{len(jobs)}] {job.patient} | {job.visit} | "
                f"{job.experiment} | {job.acquisition_type}"
            )
            try:
                decays = corrected_decays(job)
                for channel, (decay, source) in decays.items():
                    rows.append(analyze_channel(decay, source, job, channel, args))
            except Exception as error:
                traceback.print_exc()
                errors.append(
                    {
                        "job_id": job.job_id,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                if args.stop_on_error:
                    raise
            write_csv(channel_csv, rows, CHANNEL_FIELDS)
        channels = pd.DataFrame(rows, columns=CHANNEL_FIELDS)

    differences = differences_from_channels(channels)
    summary = modality_summary(channels, differences)
    visit_summary = patient_visit_summary(differences)
    difference_csv = args.output_root / "coumarin_green_blue_differences.csv"
    modality_csv = args.output_root / "coumarin_summary_by_modality.csv"
    visit_csv = args.output_root / "coumarin_summary_by_patient_visit_modality.csv"
    plot_png = args.output_root / "coumarin_difference_bars_by_modality.png"
    differences.to_csv(difference_csv, index=False)
    summary.to_csv(modality_csv, index=False)
    visit_summary.to_csv(visit_csv, index=False)
    if not differences.empty:
        plot_differences(differences, summary, plot_png, args.dpi)
    if errors:
        write_csv(args.output_root / "coumarin_errors.csv", errors, ("job_id", "error"))

    print("\nMean corrections and differences by modality:")
    print(summary.to_string(index=False))
    print(f"\nChannel table: {channel_csv}")
    print(f"Differences:   {difference_csv}")
    print(f"Modality mean: {modality_csv}")
    print(f"Visit means:   {visit_csv}")
    if not differences.empty:
        print(f"Bar plot:      {plot_png}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
