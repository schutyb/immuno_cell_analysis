from __future__ import annotations

import tempfile
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tifffile

from calibration_by_blue.calibrate_phasors import (
    PhasorJob,
    applied_deltas,
    brightest_dc_mask,
    estimate_modes,
    load_mosaic_modes,
    modes_for_job,
    phase_rotation_to_segment,
    process_job,
    rotate_phasor,
    theoretical_segment,
)
from calibration_by_blue.estimate_mosaic_deltas import (
    MOSAIC_FIELDS,
    mosaic_rows,
    write_csv,
)
from calibration_by_blue.flim_io import (
    normalize_single_detector_32_bins,
    split_green_blue,
)


def test_decay_bin_corrections_are_preserved() -> None:
    split = np.arange(2 * 3 * 31, dtype=np.float32).reshape(2, 3, 31)
    green, blue = split_green_blue(split)
    assert green.shape == (2, 3, 16)
    assert blue.shape == (2, 3, 16)
    assert np.array_equal(green[..., -1], green[..., -2])
    assert np.array_equal(blue[..., -1], blue[..., -2])

    single = normalize_single_detector_32_bins(split)
    assert single.shape == (2, 3, 32)
    assert np.array_equal(single[..., -1], single[..., -2])


def test_brightest_dc_mask_keeps_upper_35_percent() -> None:
    mean = np.arange(1, 101, dtype=np.float32).reshape(10, 10)
    real = np.full_like(mean, 0.5)
    imag = np.full_like(mean, 0.25)

    mask, threshold, valid_count = brightest_dc_mask(mean, real, imag, 35.0)

    assert valid_count == 100
    assert np.isclose(threshold, 65.35)
    assert np.count_nonzero(mask) == 35
    assert np.min(mean[mask]) == 66


def test_applied_deltas_use_protocol_specific_offsets() -> None:
    common = dict(
        patient="p1",
        visit="visit1",
        mosaic="mosaic",
        tiff_path=SimpleNamespace(),
        channels=("green", "blue"),
        tile_numbers=(1,),
    )
    split = PhasorJob(acquisition_type="Sp", **common)
    sequential = PhasorJob(acquisition_type="A1_A0", **common)

    split_deltas, _ = applied_deltas(split, {"blue": 4.0, "green": 9.0})
    sequential_deltas, _ = applied_deltas(sequential, {"blue": 4.0, "green": 9.0})

    assert split_deltas == {"blue": 4.0, "green": 6.1}
    assert sequential_deltas == {"blue": 4.0, "green": 5.55}


def test_green_without_blue_uses_own_mode() -> None:
    job = PhasorJob(
        patient="p1",
        visit="visit1",
        mosaic="A1",
        acquisition_type="A1_only",
        tiff_path=SimpleNamespace(),
        channels=("green",),
        tile_numbers=(1,),
    )

    deltas, methods = applied_deltas(job, {"green": -2.25})

    assert deltas == {"green": -2.25}
    assert methods["green"] == "own_green_mode_to_3p5_0p1ns_no_blue"


def test_phase_rotation_recovers_point_on_blue_segment() -> None:
    start, end = theoretical_segment("blue")
    target_g = start[0] + 0.65 * (end[0] - start[0])
    target_s = start[1] + 0.65 * (end[1] - start[1])
    measured_g, measured_s = rotate_phasor(
        np.asarray(target_g), np.asarray(target_s), -12.0
    )

    delta, recovered_g, recovered_s = phase_rotation_to_segment(
        float(measured_g), float(measured_s), "blue"
    )

    assert np.isclose(delta, 12.0, atol=1e-5)
    assert np.isclose(recovered_g, target_g, atol=1e-6)
    assert np.isclose(recovered_s, target_s, atol=1e-6)
    assert np.isclose(
        np.hypot(measured_g, measured_s),
        np.hypot(recovered_g, recovered_s),
        atol=1e-6,
    )


def test_small_end_to_end_tiff_is_filtered_but_not_thresholded() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source_path = root / "source.tiff"
        shape = (2, 2, 3, 16, 16)
        stack = np.empty(shape, dtype=np.float32)
        dc = np.arange(1, 257, dtype=np.float32).reshape(16, 16)
        for channel_index, channel in enumerate(("green", "blue")):
            start, end = theoretical_segment(channel)
            target_g = start[0] + 0.5 * (end[0] - start[0])
            target_s = start[1] + 0.5 * (end[1] - start[1])
            measured_g, measured_s = rotate_phasor(
                np.asarray(target_g), np.asarray(target_s), -6.0
            )
            for tile_index in range(2):
                stack[channel_index, tile_index, 0] = dc + tile_index
                stack[channel_index, tile_index, 1] = measured_g
                stack[channel_index, tile_index, 2] = measured_s
        tifffile.imwrite(source_path, stack, metadata={"axes": "CTZYX"})
        job = PhasorJob(
            patient="p1",
            visit="visit1",
            mosaic="Sp synthetic",
            acquisition_type="Sp",
            tiff_path=source_path,
            channels=("green", "blue"),
            tile_numbers=(1, 2),
        )
        args = Namespace(
            overwrite=True,
            filter_size=3,
            filter_repeat=1,
            calibration_top_dc_percent=35.0,
            plot_top_dc_percent=40.0,
            reference_bins=32,
            reference_refine_window=1,
            delta_mode_bin_width_deg=1.0,
            plot_bins=32,
            dpi=50,
            compression="zlib",
            compression_level=1,
            minimum_free_gb=0.0,
        )

        source = tifffile.memmap(source_path, mode="r", squeeze=False)
        modes, tile_records = estimate_modes(source, job, args)
        del source
        delta_csv = root / "mosaic_delta_phase.csv"
        write_csv(
            delta_csv,
            mosaic_rows(job, modes, tile_records, args.delta_mode_bin_width_deg),
            MOSAIC_FIELDS,
        )
        persisted_modes = modes_for_job(job, load_mosaic_modes(delta_csv))

        row = process_job(
            job,
            persisted_modes,
            delta_csv,
            root / "output",
            args,
        )

        assert row["status"] == "ok"
        final = tifffile.imread(row["output_tiff"], squeeze=False)
        assert final.shape == shape
        assert np.all(np.isfinite(final))
        assert np.all(final[:, :, 0] > 0)
        assert Path(row["metadata_json"]).is_file()
        assert Path(row["plot_png"]).is_file()
