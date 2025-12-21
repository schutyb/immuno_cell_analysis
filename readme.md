# immuno_cell_analysis

Pipeline for FLIM phasor–based analysis and instance-level characterization of immuno-cells in melanoma tissue.

This repository is designed as a **modular, multi-stage analysis framework**, where each stage of the pipeline is isolated, documented, and reusable.  
It was created to separate the immuno-cell FLIM analysis from other exploratory or unrelated analysis code.

Raw imaging data (FLIM stacks, mosaics, masks) and large outputs are intentionally excluded from version control.

---

## Scientific goal

The goal of this project is to:

- Analyze FLIM data acquired from melanoma tissue
- Characterize immuno-cell populations at the **instance (single-cell) level**
- Combine phasor-based lifetime information with morphological descriptors
- Enable downstream biological, statistical, and machine-learning analysis

---

## Repository structure
immuno_cell_analysis/
├── src/
│   └── immuno_cell_analysis/
│       ├── Part1_calculate_phasor/
│       │   ├── calculate_phasor.py
│       │   ├── phasor_plots.py
│       │   ├── mask_analysis.py
│       │   ├── mask_flim_parameters.py
│       │   ├── plot_final_mask.py
│       │   └── README.md
│       ├── Part2_…
│       └── …
├── README.md
└── .gitignore

Each **Part** corresponds to a well-defined stage of the analysis pipeline and has its own documentation.

---

## Pipeline overview

### Part 1 — Phasor computation and instance-level lifetime features

This stage performs:

- Phasor computation and calibration from raw FLIM mosaics
- Assembly of segmentation masks into instance masks
- Morphological filtering of segmented objects
- Extraction of per-object FLIM lifetime parameters
- Object-level filtering based on τ_phase

📄 Detailed documentation:  
➡️ `src/immuno_cell_analysis/Part1_calculate_phasor/README.md`

---

## Installation

Example environment setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy pandas tifffile matplotlib scikit-image imageio phasorpy