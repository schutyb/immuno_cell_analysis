# Final FLIM phasor calibration pipeline

This folder contains the definitive preprocessing and self-calibration flow.
The Coumarin experiments were used to choose the fixed green-from-blue
offsets; they are not part of the production pipeline.

The reproducible supporting analysis is preserved in
`../coumarin_analysis/analyze_coumarin.py`. It calculates both channel
corrections, signed green-minus-blue differences, modality averages, and bar
plots across patients and visits.

## Agreed method

1. Correct/split each raw FLIM decay and resample every temporal-bin image
   independently with the established cubic downsample/upsample method.
2. Calculate first-harmonic DC, G, and S from the resampled decay.
3. For calibration estimation only, median-filter each tile with a 7x7 kernel
   twice and retain the brightest 35% of finite positive-DC pixels (the
   threshold is the 65th percentile).
4. Estimate one modal phasor per tile and its phase rotation. Use the histogram
   mode of the successful blue tile rotations as the mosaic rotation, against
   the 0--3.5 ns blue segment.
5. Apply the blue mosaic rotation to blue. Apply:
   - `blue rotation + 2.1 degrees` to green in simultaneous split (`Sp`)
     acquisitions;
   - `blue rotation + 1.55 degrees` to green in sequential A1/A0 acquisitions.
6. If green has no usable blue acquisition, estimate its own mosaic rotation
   against the green 3.5--0.1 ns segment.
7. Apply every rotation to the original, unfiltered and unthresholded phasor
   arrays. Then median-filter DC/G/S with a 7x7 kernel twice.
8. Save the final calibrated, filtered TIFF **without intensity thresholding**.
9. Build a representative overlay plot from that saved TIFF. The plot alone
   retains the brightest 40% of DC by default.

Filtering and DC selection used to estimate calibration never replace the raw
phasors to which the calibration is applied.

## Files kept in this package

- `flim_preprocessing.py`: internal discovery, decay correction, splitting,
  and per-bin cubic resampling helpers.
- `flim_io.py`: minimal internal TIFF discovery and detector-splitting helpers.
- `calculate_corrected_phasor.py`: creates compact, unfiltered and
  unthresholded resampled phasor TIFFs.
- `estimate_mosaic_deltas.py`: filters/thresholds only for estimation, writes
  every tile delta and the histogram mode for each mosaic/channel.
- `calibrate_phasors.py`: reads the saved mosaic modes, applies calibration to
  the original phasors, and writes the final TIFF, JSON provenance, CSV
  manifest, and two-channel plot.

## Run

From the repository root:

```bash
PYTHONPATH=src .venv/bin/python \
  src/calibration_by_blue/calculate_corrected_phasor.py \
  --data-root /path/to/data_curated \
  --patients p427 p439 p449

PYTHONPATH=src .venv/bin/python \
  src/calibration_by_blue/estimate_mosaic_deltas.py \
  --data-root /path/to/data_curated \
  --patients p427 p439 p449

PYTHONPATH=src .venv/bin/python \
  src/calibration_by_blue/calibrate_phasors.py \
  --data-root /path/to/data_curated \
  --patients p427 p439 p449
```

The commands use, in order, `DATA_ROOT/corrected_phasor`,
`DATA_ROOT/mosaic_delta_phase`, and `DATA_ROOT/calibrated_filtered_phasor`.

Useful options:

```text
--overwrite
--stop-on-error
--calibration-top-dc-percent 35
--plot-top-dc-percent 40
--filter-size 7
--filter-repeat 2
```

`--calibration-top-dc-percent` belongs to the delta-estimation command;
`--overwrite`, `--plot-top-dc-percent`, and the output filter settings belong
to the final calibration. Production values are defaults and are recorded.

## Output contract

The intermediate and final TIFFs use axes `(channel, tile, component, y, x)`.
Channel order is green then blue when both exist; component order is
`dc_mean, g, s`. Tiles remain independent and are never spatially concatenated.

Each final mosaic produces:

- `*_calibrated_filtered_phasor.tiff` — calibrated and median-filtered, not
  thresholded;
- `*_calibrated_filtered_phasor.json` — applied rotations, methods, and a link
  to the delta CSV containing the per-tile calibration diagnostics;
- `*_phasor_overlay.png` — green/blue representative plot thresholded only for
  display.

The output root also contains `calibration_manifest.csv`.

## Verification

```bash
.venv/bin/ruff check \
  src/calibration_by_blue \
  tests/test_calibrate_phasors.py

MPLCONFIGDIR=/tmp/immuno_mpl_cache PYTHONPATH=src .venv/bin/python -c \
  'import runpy; n=runpy.run_path("tests/test_calibrate_phasors.py"); [f() for k,f in n.items() if k.startswith("test_")]'
```

The tests include an end-to-end synthetic TIFF and verify that the final TIFF
is filtered but is not DC-thresholded.
