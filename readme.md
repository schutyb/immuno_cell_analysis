# immuno_cell_analysis

Pipeline for FLIM phasor–based analysis, segmentation, correction, and instance-level characterization of immuno-cells in melanoma tissue.

This repository is designed as a modular and reproducible framework for:

- FLIM phasor analysis
- RGB reconstruction from FLIM decays
- immune-cell segmentation
- elastin-based correction
- ROI-level phasor filtering
- downstream quantitative cell analysis

The project is organized into independent analysis blocks so that each stage of the pipeline can be developed, validated, and reused separately.

Raw imaging data (FLIM stacks, mosaics, masks, and large outputs) are intentionally excluded from version control.

---

# Scientific goal

The main goal of this project is to:

- Analyze FLIM data acquired from melanoma tissue
- Detect and characterize immune-cell populations
- Combine phasor-based lifetime information with segmentation
- Quantify cell populations longitudinally across visits
- Enable downstream biological and statistical analysis
- Study relationships between:
  - FLIM lifetime
  - morphology
  - spatial organization
  - treatment response

---

# Repository structure

```text
immuno_cell_analysis/

├── src/
│
│   ├── segmentation/
│   │   ├── README_SEG.md
│   │   ├── build_area_phasor_cell_mask.py
│   │   ├── evaluate_final_masks_against_manual.py
│   │   ├── summarize_final_segmentation_evaluation.py
│   │   ├── analyze_segmentation_metrics_by_depth.py
│   │   ├── export_rgb_segdata_overlay_pdf.py
│   │   ├── export_rgb_manual_mask_overlay_pdf.py
│   │   ├── export_candidate_vs_final_mask_pdf.py
│   │   └── export_manual_vs_final_mask_overlay.py
│
│   ├── phasor/
│   │   ├── create_calibrated_phasor_mosaic.py
│   │   ├── phasor_calibrated_plot.py
│   │   └── ...
│
│   ├── correction/
│   │   └── ...
│
│   ├── analysis/
│   │   └── ...
│
│   └── utils/
│       ├── flim2rgb.py
│       ├── color_scales.py
|       ├── make_rgb_mosaic_pdf.py
│       └── ...
│
├── pyproject.toml
├── README.md
└── .gitignore
```

---

# Pipeline overview

The pipeline is divided into multiple logical stages.

---

# 1. FLIM → RGB reconstruction

Scripts in:

```text
src/utils/
```

Main script:

```text
flim2rgb.py
```

This stage:

- reads raw FLIM stacks
- separates green and blue detector channels
- reconstructs RGB representations from decay bins
- normalizes all tiles consistently
- reconstructs RGB mosaics

Used for:

- visualization
- annotation
- segmentation QC

---

# 2. Phasor computation and calibration

Scripts in:

```text
src/phasor/
```

Main scripts:

```text
create_calibrated_phasor_mosaic.py
phasor_calibrated_plot.py
```

This stage:

- computes first-harmonic phasors
- calibrates phasors using coumarin reference data
- creates phasor mosaics
- visualizes phasor distributions
- generates pseudocolor lifetime maps

Outputs include:

```text
DC image
G/S green detector
G/S blue detector
phasor QC plots
phasor pseudocolor images
```

---

# 3. Segmentation and phasor filtering

Scripts in:

```text
src/segmentation/
```

Main script:

```text
build_area_phasor_cell_mask.py
```

This stage combines:

```text
candidate segmentation masks
+
area filtering
+
ROI-level phasor analysis
+
GMM lifetime classification
```

to produce the final immune-cell masks.

The final method:

```text
U-Net / candidate segmentation
+
FLIM phasor filtering
```

is used for downstream biological analysis.

Detailed documentation:

```text
src/segmentation/README_SEG.md
```

---

# 4. Elastin-based correction

Scripts in:

```text
src/correction/
```

This stage is used to:

- normalize phasor shifts across visits
- compensate for acquisition variability
- use elastin as a stable tissue reference
- improve longitudinal consistency

---

# 5. Cell-level quantitative analysis

Scripts in:

```text
src/analysis/
```

This stage is intended for:

- cell density analysis
- longitudinal visit comparisons
- morphology analysis
- lifetime distributions
- phenotype association
- spatial analysis
- statistical analysis
- downstream machine learning

---

# Segmentation strategy

The segmentation strategy used in this repository is ROI-based rather than purely pixel-based.

The final masks are generated using:

```text
candidate object detection
+ area filtering
+ FLIM phasor/lifetime filtering
```

Therefore, the most important evaluation metrics are:

```text
object_precision
false_positive_percentage
relative_cell_count_error
precision
```

Dice and IoU are included as secondary overlap metrics.

---

# Installation

Example environment setup:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -U pip

pip install -e ".[dev]"
```

---

# Formatting and linting

The repository uses:

- black
- ruff

Run formatting with:

```bash
black src
ruff check src --fix
ruff format src
```

Check imports and syntax with:

```bash
python -m compileall src
```

---

# Data organization

The pipeline assumes a directory structure similar to:

```text
patients/
└── p449/
    ├── visit01/
    │   ├── Mosaic01_.../
    │   │   ├── flim/
    │   │   ├── RGB/
    │   │   ├── phasor/
    │   │   ├── SegData/
    │   │   └── ...
    │   └── ...
    └── ...
```

---

# Notes

This repository was intentionally separated from exploratory or unrelated code in order to keep the immune-cell FLIM analysis pipeline:

- modular
- reproducible
- easier to maintain
- easier to document
- easier to scale to larger datasets

The project is still under active development.