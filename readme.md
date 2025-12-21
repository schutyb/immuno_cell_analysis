# immuno_cell_analysis

Pipeline for FLIM phasor–based analysis and instance-level segmentation of immuno-cells in melanoma tissue.

This repository was created to isolate and document the immuno-cell FLIM analysis pipeline, separating it from other exploratory or unrelated analysis code.

---

## Project structure

All raw data (FLIM stacks, masks, outputs) are intentionally excluded from version control.

---

## Phase 1 — Phasor computation, masks, and per-object lifetime features

This first phase converts raw FLIM mosaics into calibrated phasor maps, builds instance masks from segmentation outputs, and computes per-object lifetime statistics.

### Expected input data layout


- Each Mosaic folder contains 16 tiles arranged in a 4×4 serpentine scan.
- Mask tiles correspond exactly to the raw mosaic tiles.

---

## Step 1 — Phasor computation and calibration  
**Script:** `calculate_phasor.py`

**What it does**
- Loads the 16 FLIM tiles per Mosaic and assembles a 4×4 mosaic.
- Converts raw data into `(T, Y, X)` format robustly.
- Computes phasor coordinates for harmonic 1 and 2.
- Uses **only the first decay window** (`T_START:T_END`, default: 0–15).
- Computes a Coumarin reference phasor (intensity-weighted center of mass).
- Calibrates phasor coordinates using:
  - Frequency = **80 MHz**
  - Reference lifetime = **2.5 ns**
- Saves both uncalibrated and calibrated phasor maps.

**Outputs (per Mosaic)**


**CYX channel layout**
| Channel | Description |
|------|------------|
| 0 | Mean intensity |
| 1 | g (H1) |
| 2 | s (H1) |
| 3 | Modulation (H1) |
| 4 | Phase (H1, rad) |
| 5 | g (H2) |
| 6 | s (H2) |
| 7 | Modulation (H2) |
| 8 | Phase (H2, rad) |

---

## Step 2 — Median filtering and QC plots  
**Script:** `phasor_plots.py`

**What it does**
- Recursively finds all calibrated phasor TIFFs.
- Applies median filtering (`phasor_filter_median`) to:
  - Mean intensity
  - Real/imaginary phasor components
- Recomputes polar coordinates after filtering.
- Generates QC plots.

**Outputs**
- `phasor_filtered_median_CYX.tif`
- `mean_image.jpg` (grayscale)
- `mean_histogram.jpg`
- `phasor_H1.jpg` (phasor density plot using `PhasorPlot.hist2d`)

---

## Step 3 — Mask mosaic assembly and instance segmentation  
**Script:** `mask_analysis.py`

**What it does**
- Loads the 16 mask tiles per Mosaic (alphabetical order assumed).
- Assembles a 4×4 mask mosaic using the same scan order as FLIM data.
- Converts to binary mask.
- Labels connected components → instance mask.
- Filters objects by minimum equivalent diameter (default: **8 px**).
- Computes morphological features per instance.

**Morphological features computed**
- Area
- Equivalent diameter
- Perimeter
- Circularity
- Eccentricity
- Major axis length
- Minor axis length
- Major/minor axis ratio
- Centroid (x, y)

**Outputs (inside each mask folder)**


---

## Step 4 — Per-instance lifetime parameters and τ filtering  
**Script:** `mask_flim_parameters.py`

**What it does**
- Loads:
  - Instance mask
  - Morphology CSV
  - Calibrated phasor maps
- Computes apparent lifetimes from polar coordinates:
  - τ_phase
  - τ_modulation
- Computes per-instance mean values:
  - g, s
  - phase, modulation
  - τ_phase, τ_modulation
- Filters objects based on **mean τ_phase** (default: 0–12 ns).
- Relabels instance masks after filtering.

**Outputs**

---

## Step 5 — Visual QC of final instance masks  
**Script:** `plot_final_mask.py`

**What it does**
- User provides a single TIFF path inside a `mask_*` folder.
- Automatically loads:
  - Diameter-filtered instance mask
  - Diameter + τ-filtered instance mask
- Saves a side-by-side visualization for QC.

**Output**


---

## Installation

Example environment setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy pandas tifffile matplotlib scikit-image imageio phasorpy