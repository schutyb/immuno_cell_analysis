# Coumarin reference analysis

This package preserves the analysis used to select the green-from-blue phase
offsets. It is evidence and QC for the production calibration; it is not run
for every tissue mosaic.

The script reads all Coumarin acquisitions across patients and visits,
recognizes simultaneous `Sp` and sequential `A1/A0` modalities, calculates a
phase correction independently for green and blue, pairs the channels, and
calculates:

```text
signed difference = green correction - blue correction
positive magnitude = abs(signed difference)
```

The signed value matches the historical Coumarin bar plots. The production
pipeline uses `abs(mean signed difference)` as the positive green-from-blue
offset, currently rounded to 2.1 degrees for Sp and 1.55 degrees for A1/A0.
The summary also reports the mean of the individual absolute differences as a
separate diagnostic; that is not the production offset.

## Raw analysis

```bash
PYTHONPATH=src .venv/bin/python \
  src/coumarin_analysis/analyze_coumarin.py \
  --data-root /path/to/data_curated
```

Defaults reproduce the established analysis: 2.5 ns Coumarin at 80 MHz,
50-pixel cubic spatial resampling, median 5x5 repeated twice, and removal of
the lowest 25% of positive DC.

## Rebuild summaries from an existing channel table

```bash
PYTHONPATH=src .venv/bin/python \
  src/coumarin_analysis/analyze_coumarin.py \
  --channel-csv /path/to/coumarin_channel_calibration.csv \
  --output-root /path/to/new_summary
```

Historical tables using `phase_shift_deg` are accepted automatically.

## Outputs

- `coumarin_channel_calibration.csv`: correction for every channel acquisition;
- `coumarin_green_blue_differences.csv`: paired green/blue differences;
- `coumarin_summary_by_modality.csv`: averages for Sp and A1/A0;
- `coumarin_summary_by_patient_visit_modality.csv`: averages for every
  patient/visit/modality case;
- `coumarin_difference_bars_by_modality.png`: signed bar plots with modality
  means and positive magnitudes;
- `coumarin_errors.csv`: unreadable or invalid acquisitions, when present.
