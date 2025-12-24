# Part 2 — Phasor-based clustering and morphology statistics of immuno-cells

This module performs **phasor-domain clustering and downstream morphological statistical analysis** on segmented immuno-cells derived from FLIM phasor analysis.

It operates **exclusively on per-instance CSV outputs generated in Part 1**, where objects have already been:
- segmented,
- filtered by minimum equivalent diameter,
- filtered by lifetime range (τ_phase ∈ [0, 12] ns).

No raw FLIM data or pixel-level images are used in this stage.

---

## Overview of Part 2

**Inputs (from Part 1):**
- `mask_instances_features_minEqDiam8px_tau0-12_ONLY_cells.csv`
- stored under:

data_patient_449/phasor_cluster_out/
visit_XX/
MosaicYY_…/
*_ONLY_cells.csv

**Outputs (Part 2):**
1. Phasor-domain clustering (GMM) separating:
 - melanina
 - cells
 - elastina
2. Cell-only CSVs (after phasor classification)
3. High-quality phasor plots
4. Morphological statistics:
 - quantized violin plots
 - mean / median / modes
 - trends across visits

---

## Step 2.1 — Phasor clustering of segmented objects  
**Script:** `phasor_clustering.py`

### What this script does

For each `*_ONLY_cells.csv`:

1. Loads per-instance **mean phasor coordinates**:
 - `g_mean`
 - `s_mean`

2. Performs **Gaussian Mixture Model (GMM)** clustering in phasor space.

3. Automatically determines the number of clusters:
 - **3 clusters** by default:
   - melanin
   - cell
   - elastin
 - **2 clusters for visit 4** (melanin absent):
   - cell
   - elastin

4. Automatically assigns biological labels to clusters:
 - **3 clusters** → sorted by **phasor angle**:
   - smallest angle → melanin
   - middle angle → cell
   - largest angle → elastin
 - **2 clusters** → sorted by **lifetime proxy**:
   - lower τ → cell
   - higher τ → elastin

5. Saves:
 - a CSV containing **only objects classified as cells**
 - a high-quality phasor scatter plot (JPG)
 - a summary CSV with per-file counts

### Outputs

For each input CSV:

- `{name}_ONLY_cells.csv`
- `{name}_phasor_segmented_phasorpy.jpg`

Global summary:
- `summary_counts.csv`

All outputs are saved under:

phasor_cluster_out/

---

## Step 2.2 — Morphological statistics on phasor-selected cells  
**Script:** `morphology_stats_cells.py`

### What this script does

This script analyzes **only the objects classified as cells by phasor clustering**.

1. Automatically discovers all:

*_ONLY_cells.csv

under:

data_patient_449/phasor_cluster_out

2. Groups objects by visit (`visit_01`, `visit_02`, ...).

3. For each visit and each morphological feature:
- Quantizes values to reflect effective measurement resolution
- Estimates distributions using KDE
- Extracts:
  - mean
  - median
  - up to 3 modes

4. Generates **publication-quality visualizations**:
- Quantized violin plots with:
  - IQR boxes
  - whiskers (5–95%)
  - short horizontal markers ("rayitas") for:
    - mean (red)
    - median (green)
    - modes (blue)
- Trend plots across visits:
  - mean vs visit
  - median vs visit
  - modes vs visit

### Morphological features analyzed

- `area_px2`
- `equivalent_diameter_px`
- `circularity`
- `major_axis_length_px`
- `minor_axis_length_px`
- `perimeter_px`

### Outputs

All figures are saved as **high-resolution JPGs** in:

phasor_cluster_out/plots_cells_violin_quantized/

Files include:
- `violin_<feature>_by_visit_quantized_multimode_rays.jpg`
- `trend_mean_<feature>_vs_visit.jpg`
- `trend_median_<feature>_vs_visit.jpg`
- `trend_modes_<feature>_vs_visit.jpg`

---

## Design principles

- **Strict separation of pipeline stages**
- **No recomputation of segmentation or lifetimes**
- **Object-level analysis only**
- **Deterministic clustering**
- **Fully automated visit discovery**
- **Publication-ready figures**

---

## Dependencies

```bash
pip install numpy pandas matplotlib scipy scikit-learn phasorpy


