# immuno_cell_analysis

Reproducible FLIM phasor preprocessing and blue-derived self-calibration.

The production pipeline is intentionally small. Historical calibration,
image-composition, and segmentation experiments have been removed now that the
final calibration method is fixed. A clean Coumarin reference analysis is kept
to document and reproduce the modality-specific offsets.

## Final method

1. Read the raw FLIM tiles and correct/split the decay bins.
2. Resample every temporal-bin image independently with the established cubic
   downsample/upsample method.
3. Calculate first-harmonic DC, G, and S from the resampled decays.
4. For calibration estimation only, apply a 7x7 median filter twice and retain
   the brightest 35% of finite positive-DC pixels in each tile.
5. Estimate the blue rotation for each tile against the 0--3.5 ns segment and
   use the histogram mode of the tile rotations as the mosaic rotation.
6. Calibrate green using:
   - blue rotation +2.1 degrees for simultaneous split (`Sp`) acquisitions;
   - blue rotation +1.55 degrees for sequential A1/A0 acquisitions;
   - its own mode against the 3.5--0.1 ns green segment when blue is absent.
7. Apply calibration to the original unfiltered and unthresholded G/S arrays.
8. Median-filter the calibrated DC/G/S maps with a 7x7 kernel twice and save
   them without DC thresholding.
9. Create a representative green/blue phasor plot from the saved TIFF. The
   plot retains the brightest 40% of DC for visualization only.

## Project structure

```text
src/
├── calibration_by_blue/
│   ├── calculate_corrected_phasor.py  # resampling and raw phasors
│   ├── estimate_mosaic_deltas.py      # tile deltas and mosaic modes
│   ├── calibrate_phasors.py           # final calibration, TIFF and plot
│   ├── flim_preprocessing.py          # resampling helpers
│   ├── flim_io.py                     # FLIM discovery and bin correction
│   └── README.md                      # detailed contract and options
├── coumarin_analysis/
│   ├── analyze_coumarin.py            # reference corrections and differences
│   └── README.md
└── utils/                             # retained shared RGB utilities

tests/
└── test_calibrate_phasors.py
```

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

The first stage defaults to `DATA_ROOT/corrected_phasor`. The second saves
tile deltas and mosaic modes under `DATA_ROOT/mosaic_delta_phase`. The third
reads those modes and writes the definitive products to
`DATA_ROOT/calibrated_filtered_phasor`.

## Final products

The delta-estimation stage produces:

- `tile_delta_phase.csv`: delta and diagnostics for every channel/tile;
- `mosaic_delta_phase.csv`: histogram mode for every mosaic/channel;
- `run_metadata.json`: all estimation parameters.

The final calibration stage produces:

For every mosaic:

- `*_calibrated_filtered_phasor.tiff`: calibrated and median-filtered, not
  thresholded;
- `*_calibrated_filtered_phasor.json`: parameters, applied rotations,
  provenance, and per-tile diagnostics;
- `*_phasor_overlay.png`: representative green/blue plot, thresholded only for
  visualization.

The output root also contains `calibration_manifest.csv`.

See `src/calibration_by_blue/README.md` for the complete array layout and CLI
options.

## Verification

```bash
.venv/bin/ruff check src/calibration_by_blue tests

MPLCONFIGDIR=/tmp/immuno_mpl_cache PYTHONPATH=src .venv/bin/python -c \
  'import runpy; from pathlib import Path; [[f() for k,f in runpy.run_path(str(p)).items() if k.startswith("test_")] for p in Path("tests").glob("test_*.py")]'
```

The test suite covers detector-bin correction, DC selection, modality-specific
offsets, green-only fallback, phase geometry, and a compressed end-to-end TIFF.
