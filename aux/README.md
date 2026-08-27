# Auxiliary median-filter comparison

`compare_median_filters.py` compares the production median filter (7x7,
repeated twice) against 9x9 once and 11x11 once. It is isolated from the main
pipeline and never overwrites production phasors.

Defaults:

- patient `p449`;
- `visit01/Mosaic03_4x4_FOV600_z110_32Sp`;
- `visit04/Mosaic04_4x4_FOV600_z135_32Sp`;
- production cubic resampling, 1200→1000→1200;
- first-harmonic phasor;
- brightest 35% DC for plotting only.

Run from the repository root:

```bash
export MPLCONFIGDIR=/private/tmp/immuno_mpl_cache

PYTHONPATH=src .venv/bin/python \
  aux/compare_median_filters.py
```

For a quick one-tile validation:

```bash
PYTHONPATH=src .venv/bin/python \
  aux/compare_median_filters.py \
  --max-tiles 1 \
  --dpi 100 \
  --histogram-bins 128
```

Outputs are written to `aux/median_filter_comparison_results/`:

- one combined comparison plot per mosaic;
- one two-channel phasor plot per mosaic/filter;
- `filter_timings.csv` with total time, seconds/tile, and speedup;
- `filter_similarity_vs_7x7x2.csv` with G/S RMSE and phase differences;
- `mosaic_processing_times.csv` with resampling and phasor calculation times.

Only `phasor_filter_median` is included in the reported filter time. Raw TIFF
reading, resampling, phasor calculation, threshold selection, histogramming,
and plotting are excluded, ensuring a fair filter comparison.

## Calibrated phasor histograms with PhasorPlot

`plot_calibrated_phasors_phasorplot.py` reads the final calibrated and filtered
TIFFs, selects the brightest 35% of DC independently in every tile for display
only, and plots each channel with `PhasorPlot.hist2d`. It uses Matplotlib's
`RdYlGn_r` density colormap by default, writes one PNG per mosaic, and produces
one PDF containing every patient and visit.

```bash
MPLCONFIGDIR=/tmp/immuno_mpl_cache \
XDG_CACHE_HOME=/tmp/immuno_cache \
PYTHONPATH=src .venv/bin/python \
  aux/plot_calibrated_phasors_phasorplot.py
```

To hide sparse histogram bins containing fewer than 20 pixels while preserving
the same 35% DC selection and source TIFFs:

```bash
MPLCONFIGDIR=/tmp/immuno_mpl_cache \
XDG_CACHE_HOME=/tmp/immuno_cache \
PYTHONPATH=src .venv/bin/python \
  aux/plot_calibrated_phasors_phasorplot.py \
  --histogram-min-count 20 \
  --output-root /path/to/phasorplot_hist2d_top35_cmin20
```

Exact subsets can be plotted without regenerating other mosaics by combining
`--patients`, `--visits`, and `--mosaics`.

Add `--show-calibration-lines` to overlay, in each channel panel, the same
lifetime segment used by the production calibration: blue 0-3.5 ns and green
3.5-0.1 ns. This changes only the visualization and writes a separately named
PDF and PNG files, leaving the calibrated TIFFs untouched.

## Phase-colored RGB mosaics

`create_phase_rgb_mosaics.py` assembles the calibrated tiles into their declared
N x M grid, colors green-channel phase with `reds_to_greens`, colors blue-channel
phase with `blues_to_greens`, and creates a dual-channel RGB composition. The
composition averages the two phase colors using normalized DC as per-pixel
confidence and uses their union as brightness. This avoids the spatial blur
that a convolution would introduce.

For every mosaic it saves full-resolution green, blue, and composite PNGs, plus
a three-panel summary PNG. One final PDF contains the three-panel page for all
patients and visits. If blue is unavailable, the script explicitly marks that
panel and uses green alone for the composite.

```bash
PYTHONPATH=src .venv/bin/python \
  aux/create_phase_rgb_mosaics.py \
  --input-root /path/to/calibrated_filtered_phasor \
  --output-root /path/to/phase_rgb_mosaics
```

The defaults use a 0-55 degree phase range and phase gamma 0.6. Brightness is
normalized in two stages: each tile receives a bounded gain that aligns its
median DC with the mosaic median, then one percentile 1-99 range is applied to
the entire mosaic with intensity gamma 0.7. This prevents isolated bright
structures from making the rest of their tile artificially dark. These
operations affect only the RGB visualization; the calibrated TIFFs are never
rewritten. Use `--normalization-mode tile --intensity-gamma 0.99` to reproduce
the earlier independent per-tile display scaling.
