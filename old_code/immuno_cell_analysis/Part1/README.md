# Part 1 — Structure-level FLIM & Phasor Analysis

This part of the pipeline processes raw FLIM data and binary masks to obtain
**filtered structures**, **per-structure quantitative features**, and
**phasor-based biological classification**.

The output of Part 1 is a **structure-level dataset** ready for downstream
analysis and modeling.

---

## Execution order

Run the following scripts **in order**:

### 1.a. `compute_phasor_data.py`
Compute per-pixel phasor and lifetime maps from raw FLIM data.
- **Inputs:** raw FLIM TIFF, Coumarin reference
- **Output:** `phasor_data_CYX.tif`

---

### 1.b. `phasor_plot.py` *(QC step)*
Visualize GREEN and BLUE phasor distributions for quality control.
- **Input:** `phasor_data_CYX.tif`
- **Output:** phasor plot PNG

---

### 2. `mask_filter_area.py`
Filter initial binary masks by minimum area to remove small objects.
- **Input:** raw binary mask
- **Output:** area-filtered mask

---

### 3. `mask_area_tau_phase.py`
Filter structures using mean **tau_phase** (GREEN channel).
- **Inputs:** `phasor_data_CYX.tif`, area-filtered mask
- **Output:** final mask filtered by area and lifetime

---

### 4. `extract_structure_features_to_csv.py`
Extract per-structure morphology and FLIM/phasor features.
- **Inputs:** final mask, `phasor_data_CYX.tif`
- **Outputs:**
  - `structure_features.csv`
  - instance mask (`uint32` TIFF + PNG preview)

---

### 5. `classify_structures_gmm_phasor.py`
Cluster structures in phasor space using GMM and assign biological classes
(e.g. elastin, cell, melanin).
- **Input:** `structure_features.csv`
- **Output:** classified CSV and segmented phasor plot

---

## Notes
- All scripts assume **consistent image dimensions** between masks and phasor data.
- Step 5 is optional and intended for visual QC only.
- The final output of Part 1 is a **structure-level classified dataset**
  suitable for downstream analysis (Part 2).