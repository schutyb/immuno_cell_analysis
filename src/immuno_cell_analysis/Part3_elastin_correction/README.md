# Part 3 — Elastin-based correction + phase/mod calibration (QC)

This phase performs an **elastin-based harmonization across visits** to reduce inter-visit shifts in phasor space, using **per-instance mean phasor features** already labeled in Part 2 (melanin / cell / elastin).

We use elastin as the **reference population** to:
1) **Distribution-correct** (z-score remap) the phasor coordinates across visits.
2) Apply a **final calibration in (phase, modulation)** space, then convert back to **(g, s)**.
3) Generate QC plots to verify the alignment.

> **Input expectation:** CSVs classified by Part 2 (per mosaic) containing at least:
- `g_mean`, `s_mean`
- `modulation_mean`, `phase_mean_rad`
- `phasor_class_name` (labels: `elastin`, `cell`, `melanin` — note: visit_03 and visit_04 may not contain melanin)

---

## Scripts included

### 1) `elastin_phase_mod_correction.py`
**Goal:** Apply elastin-based correction and final phase/mod calibration.

**What it does**
- Recursively finds Part 2 outputs:
  `*_phasor_classified.csv`
- For each visit/mosaic CSV:
  - Uses **elastin rows** to compute per-file elastin stats:
    - `mu_G`, `sigma_G` from `g_mean`
    - `mu_S`, `sigma_S` from `s_mean`
  - Computes **global elastin averages** (across all files with enough elastin).
  - Applies distribution correction to all valid points:
    - `g_mean_corr`, `s_mean_corr`
  - Uses **elastin only** to compute final calibration constants between ORIGINAL and CORRECTED:
    - `Δphase` (circular mean difference)
    - `Mfac` (modulation scaling)
  - Applies final calibration to ORIGINAL points (all classes):
    - `g_mean_final`, `s_mean_final`

**Outputs (mirrors input directory structure under OUT_ROOT)**
Per input CSV:
- `<stem>_phasor_classified_corrected.csv`
- `<stem>_phasor_classified_corrected_calibrated.csv`

Global summary tables:
- `elastin_correction_params.csv`
- `final_phase_mod_calibration_params.csv`

**Key output columns added**
- `g_mean_corr`, `s_mean_corr`
- `g_mean_final`, `s_mean_final`

---

### 2) `plot_elastin_centroids_only.py`
**Goal:** QC elastin centroids across visits (Original vs Corrected) + global centroid-of-centroids.

**What it does**
- Loads corrected outputs from Part 3:
  `*_phasor_classified_corrected_calibrated.csv`
- For each visit:
  - Computes elastin centroid using **Original**: `(g_mean, s_mean)`
  - Computes elastin centroid using **Corrected**: `(g_mean_corr, s_mean_corr)`
- Computes:
  - **Global mean of per-visit original centroids**
  - **Global mean of per-visit corrected centroids**
- Generates a QC plot:
  - Per-visit original centroids (blue)
  - Per-visit corrected centroids (green)
  - Global mean original (orange)
  - Global mean corrected (red)
  - Includes a zoom-in panel around the centroid region (inset)

**Outputs**
- `elastin_centroids_only_table.csv` (per-visit centroids + global means)
- `phasor_elastin_centroids_only_zoom.jpg` (QC plot)

---

### 3) `corrected_phasor_plot.py`
**Goal:** QC phasor distributions by class (Original vs Final), per visit.

**What it does**
- Loads Part 3 calibrated CSVs:
  `*_phasor_classified_corrected_calibrated.csv`
- For each visit:
  - Plots ORIGINAL points `(g_mean, s_mean)` by class:
    - elastin = green
    - cell = orange
    - melanin = blue (if present)
  - Plots FINAL points `(g_mean_final, s_mean_final)` by class with different palette:
    - elastin = gray
    - cell = red
    - melanin = purple (if present)
- Saves **600 dpi** JPGs (for heavy zooming).

**Outputs**
- `visit_XX_phasor_ORIG_vs_FINAL.jpg` (one per visit)
- Optionally: a global plot combining all visits (FINAL only), depending on your script configuration.

---

## Folder / IO conventions

### Input root (from Part 2)
Part 2 outputs typically live under:

data_patient_449/phasor_cluster_out/visit_XX/MosaicYY…/
mask_instances_features_minEqDiam8px_tau0-12_phasor_classified.csv

### Output root (Part 3)
Part 3 writes to:

data_patient_449/phasor_cluster_out/part3_elastin_phase_mod_correction_out/
visit_XX/MosaicYY…/
…_corrected.csv
…_corrected_calibrated.csv
elastin_correction_params.csv
final_phase_mod_calibration_params.csv

---

## How to run

Activate your environment first:
```bash
source .venv/bin/activate
# or
source .mel-venv/bin/activate

Step 1 — Elastin correction + final calibration
python elastin_phase_mod_correction.py

Step 2 — QC elastin centroids (Original vs Corrected + global means)
python plot_elastin_centroids_only.py

Step 3 — QC class phasor plots (Original vs Final)
python corrected_phasor_plot.py
