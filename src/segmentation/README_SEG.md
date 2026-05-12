# Segmentation pipeline

This folder contains the scripts used to generate, inspect, and evaluate the final immune-cell segmentation masks.
Based on the candidate mask previously segmented with a general U-Net Model trained in the NLOML Lab 
at the Beckman Laser Institute, Irvine, CA. The model was trained with Lifetime Imaging data of the skin cells 
and cell-like structures. 

---

# Pipeline overview

```text
Candidate masks / SegData
        ↓
Area filtering
        ↓
ROI phasor extraction
        ↓
GMM lifetime classification
        ↓
Final cell masks
        ↓
QC overlays
        ↓
Evaluation against manual masks
        ↓
Summary plots and depth analysis
```

---

# Folder structure

```text
src/segmentation/

├── build_area_phasor_cell_mask.py
├── evaluate_final_masks_against_manual.py
├── summarize_final_segmentation_evaluation.py
├── analyze_segmentation_metrics_by_depth.py
│
├── export_rgb_segdata_overlay_pdf.py
├── export_rgb_manual_mask_overlay_pdf.py
├── export_candidate_vs_final_mask_pdf.py
├── export_manual_vs_final_mask_overlay.py
│
└── README_SEG.md
```

---

# 1. Build final cell masks

## Script

```bash
python -m src.segmentation.build_area_phasor_cell_mask
```

---

## Purpose

This is the core segmentation script.

It takes:

```text
- candidate masks from SegData/
- calibrated phasor mosaics from phasor/
```

and generates the final immune-cell masks using:

```text
candidate segmentation
+ area filtering
+ ROI-level phasor analysis
+ GMM lifetime classification
```

---

## Input

```text
<mosaic>/SegData/
<mosaic>/phasor/
```

---

## Output

```text
<mosaic>/final_masks_area_phasor_gmm/
```

with:

```text
<mosaic>_cell_mask_final.tif
<mosaic>_mask_lifetime_classes.tif
<mosaic>_roi_area_phasor_gmm.csv
<mosaic>_gmm_cluster_summary.csv
<mosaic>_elastin_cm.csv
tiles/
```

---

## Biological classes

### 2-component GMM

```text
cell
elastin
```

### 3-component GMM

```text
melanin
cell
elastin
```

Clusters are assigned automatically based on mean phase:

```text
lowest phase      → melanin
intermediate      → cell
highest phase     → elastin
```

---

## Important parameters

Edit inside the script:

```python
PATIENT_DIR = Path(...)

MIN_AREA_PX = 20

CELL_PROB_THRESHOLD = 0.5

VISIT_GMM_COMPONENTS = {
    "visit01": 3,
    "visit02": 3,
    "visit03": 3,
    "visit04": 2,
}
```

---

# 2. QC: RGB + candidate SegData masks

## Script

```bash
python -m src.segmentation.export_rgb_segdata_overlay_pdf
```

---

## Purpose

Creates PDFs showing:

```text
RGB tile
+
candidate SegData mask
```

Used to inspect the original candidate segmentation before phasor filtering.

---

## Input

```text
<mosaic>/RGB/
<mosaic>/SegData/
```

---

## Output

```text
<patient>/QC_rgb_segdata_overlay_PDFs/
```

---

# 3. QC: RGB + manual masks

## Script

```bash
python -m src.segmentation.export_rgb_manual_mask_overlay_pdf
```

---

## Purpose

Creates PDFs showing:

```text
RGB tile
+
manual expert mask
```

Used to verify manual annotations and RGB alignment.

---

## Input

```text
<mosaic>/RGB/
<mosaic>/random_forest/mask/
```

---

## Output

```text
<patient>/QC_manual_mask_rgb_overlay_PDFs/
```

---

# 4. QC: candidate masks vs final masks

## Script

```bash
python -m src.segmentation.export_candidate_vs_final_mask_pdf
```

---

## Purpose

Compares:

```text
original candidate masks
vs
final area + phasor/GMM masks
```

Each PDF page contains:

```text
candidate mask
RGB + candidate overlay
final mask
RGB + final overlay
```

Used to inspect what was removed by the FLIM filtering.

---

## Input

```text
<mosaic>/RGB/
<mosaic>/SegData/
<mosaic>/final_masks_area_phasor_gmm/tiles/
```

---

## Output

```text
<patient>/QC_candidate_vs_final_mask_PDFs/
```

---

# 5. QC: manual masks vs final masks

## Script

```bash
python -m src.segmentation.export_manual_vs_final_mask_overlay
```

---

## Purpose

Creates mask-only overlays between:

```text
manual masks
vs
final predicted masks
```

Color convention:

```text
green   = manual only
red     = prediction only
yellow  = overlap
```

Used for visual agreement inspection.

---

## Input

```text
<mosaic>/random_forest/mask/
<mosaic>/final_masks_area_phasor_gmm/tiles/
```

---

## Output

```text
<patient>/segmentation_evaluation/manual_vs_final_mask_overlay_masks/
```

---

# 6. Evaluate final masks against manual masks

## Script

```bash
python -m src.segmentation.evaluate_final_masks_against_manual
```

---

## Purpose

Computes quantitative segmentation metrics comparing:

```text
manual masks
vs
final area + phasor/GMM masks
```

---

## Main metrics

```text
precision
object_precision
fp_objects
false_positive_percentage
relative_cell_count_error
dice
iou
```

---

## Input

```text
<mosaic>/random_forest/mask/
<mosaic>/final_masks_area_phasor_gmm/tiles/
```

---

## Output

```text
<patient>/segmentation_evaluation/final_area_phasor_gmm/
```

with:

```text
segmentation_key_metrics_by_tile.csv
segmentation_metrics_summary.csv
final_segmentation_key_metrics_panel.png
final_segmentation_object_precision_and_fp_percentage.png
final_segmentation_performance_cards.png
README_segmentation_evaluation.txt
```

---

# 7. Summarize final segmentation evaluation

## Script

```bash
python -m src.segmentation.summarize_final_segmentation_evaluation
```

---

## Purpose

Reads the tile-level evaluation CSV and creates a compact final summary.

Used for manuscript-ready reporting.

---

## Input

```text
segmentation_key_metrics_by_tile.csv
```

---

## Output

```text
final_segmentation_evaluation/
```

with:

```text
summary CSVs
summary plots
evaluation explanation text
```

---

# 8. Analyze segmentation metrics by depth

## Script

```bash
python -m src.segmentation.analyze_segmentation_metrics_by_depth
```

---

## Purpose

Extracts imaging depth from mosaic names such as:

```text
Mosaic03_4x4_FOV600_z110_32Sp
```

and analyzes segmentation performance as a function of depth.

---

## Output

```text
<patient>/segmentation_evaluation/depth_analysis/
```

with:

```text
tile_metrics_with_depth.csv
scatter_*_vs_depth.png
boxplot_*_by_depth_group.png
summary_panel_by_depth.png
depth_group_summary.csv
```

---

# Recommended execution order

```bash
# 1. Build final masks
python -m src.segmentation.build_area_phasor_cell_mask

# 2. QC candidate masks
python -m src.segmentation.export_rgb_segdata_overlay_pdf

# 3. QC manual masks
python -m src.segmentation.export_rgb_manual_mask_overlay_pdf

# 4. QC candidate vs final masks
python -m src.segmentation.export_candidate_vs_final_mask_pdf

# 5. QC manual vs final masks
python -m src.segmentation.export_manual_vs_final_mask_overlay

# 6. Quantitative evaluation
python -m src.segmentation.evaluate_final_masks_against_manual

# 7. Final evaluation summary
python -m src.segmentation.summarize_final_segmentation_evaluation

# 8. Depth analysis
python -m src.segmentation.analyze_segmentation_metrics_by_depth
```

---

# Current assumptions

The current scripts assume:

```text
RGB images:
<mosaic>/RGB/

Candidate masks:
<mosaic>/SegData/

Manual masks:
<mosaic>/random_forest/mask/

Final masks:
<mosaic>/final_masks_area_phasor_gmm/tiles/
```

---

# Notes

The segmentation strategy is ROI-based rather than purely pixel-based.

The final method combines:

```text
candidate object detection
+ Cadidate mask previously created with U-Net 
+ area filtering
+ FLIM phasor/lifetime filtering
```

Therefore, the most biologically relevant metrics are:

```text
object_precision
false_positive_percentage
relative_cell_count_error
precision
```

Dice and IoU are secondary overlap metrics and are less central for the biological interpretation of immune-cell quantification.
