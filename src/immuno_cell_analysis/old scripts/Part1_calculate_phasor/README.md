---

# 📄 README DE LA PARTE 1

👉 **Archivo:**  
`src/immuno_cell_analysis/Part1_calculate_phasor/README.md`

Copiá **todo esto**:

```markdown
# Part 1 — Phasor computation, masks, and per-object lifetime features

This module implements the first stage of the immuno-cell FLIM analysis pipeline.

The goal of Part 1 is to convert raw FLIM mosaics into calibrated phasor maps, build instance-level segmentation masks, and compute per-object lifetime and morphological parameters.

---

## Expected input data layout

- Data are organized by `visit_*` folders.
- Each visit contains one or more `Mosaic*` folders.
- Each Mosaic folder contains **16 FLIM tiles** arranged in a **4×4 serpentine scan order**.
- Each Mosaic folder also contains a corresponding `mask_*` folder with **16 segmentation mask tiles**, matching the raw FLIM tiles.

Raw data are not versioned in this repository.

---

## Step 1 — Phasor computation and calibration  
**Script:** `calculate_phasor.py`

### What it does
- Loads the 16 FLIM tiles per Mosaic and assembles a 4×4 mosaic.
- Robustly converts raw data into `(T, Y, X)` format.
- Computes phasor coordinates for harmonic 1 and 2.
- Uses **only the first decay window** (`T_START:T_END`, default: bins 0–15).
- Computes a Coumarin reference phasor using an intensity-weighted center of mass.
- Calibrates phasor coordinates using:
  - Laser frequency = **80 MHz**
  - Reference lifetime = **2.5 ns**
- Saves both uncalibrated and calibrated phasor maps.

### Outputs (per Mosaic)

- `phasor_uncalibrated_CYX.tif`
- `phasor_calibrated_CYX.tif`

### CYX channel layout

| Channel | Description |
|-------|------------|
| 0 | Mean intensity |
| 1 | g (H1) |
| 2 | s (H1) |
| 3 | Modulation (H1) |
| 4 | Phase (H1, radians) |
| 5 | g (H2) |
| 6 | s (H2) |
| 7 | Modulation (H2) |
| 8 | Phase (H2, radians) |

---

## Step 2 — Median filtering and QC plots  
**Script:** `phasor_plots.py`

### What it does
- Recursively finds all calibrated phasor TIFFs.
- Applies median filtering (`phasor_filter_median`) to:
  - Mean intensity
  - Real and imaginary phasor components
- Recomputes polar coordinates after filtering.
- Generates quality-control visualizations.

### Outputs
- `phasor_filtered_median_CYX.tif`
- `mean_image.jpg` (grayscale)
- `mean_histogram.jpg`
- `phasor_H1.jpg` (phasor density plot using `PhasorPlot.hist2d`)

---

## Step 3 — Mask mosaic assembly and instance segmentation  
**Script:** `mask_analysis.py`

### What it does
- Loads the 16 mask tiles per Mosaic (alphabetical order assumed).
- Assembles a 4×4 mask mosaic using the same scan order as the FLIM data.
- Converts the mosaic into a binary mask.
- Labels connected components to generate an instance mask.
- Filters objects by **minimum equivalent diameter** (default: 8 px).
- Computes morphological features per instance.

### Morphological features computed
- Area
- Equivalent diameter
- Perimeter
- Circularity
- Eccentricity
- Major axis length
- Minor axis length
- Major/minor axis ratio
- Centroid (x, y)

### Outputs (inside each `mask_*` folder)
- `mask_mosaic_binary.tif`
- `mask_instances_minEqDiam8px.tif`
- `mask_instances_features_minEqDiam8px.csv`

---

## Step 4 — Per-instance lifetime parameters and τ filtering  
**Script:** `mask_flim_parameters.py`

### What it does
- Loads:
  - Instance mask
  - Morphological features CSV
  - Calibrated phasor maps
- Computes apparent lifetimes from polar coordinates:
  - τ_phase
  - τ_modulation
- Computes **per-instance mean values** of:
  - g, s
  - phase, modulation
  - τ_phase, τ_modulation
- Filters objects based on **mean τ_phase** (default range: 0–12 ns).
- Relabels instance masks after filtering.

### Outputs
- `mask_instances_features_minEqDiam8px_plusLifetime.csv`
- `mask_instances_features_minEqDiam8px_tau0-12.csv`
- `mask_instances_minEqDiam8px_tau0-12.tif`

---

## Step 5 — Visual QC of final instance masks  
**Script:** `plot_final_mask.py`

### What it does
- User provides a single TIFF path inside a `mask_*` folder.
- Automatically loads:
  - Diameter-filtered instance mask
  - Diameter + τ-phase–filtered instance mask
- Generates a side-by-side visualization for quality control.

### Output
- `QC_instances_diam_vs_tau.png`

---

## Summary

At the end of Part 1, each Mosaic is associated with:
- Calibrated phasor maps
- Instance-level segmentation masks
- Morphological descriptors
- Per-object FLIM lifetime parameters
- Filtered populations based on biologically meaningful criteria

These outputs serve as the foundation for downstream analysis stages.
