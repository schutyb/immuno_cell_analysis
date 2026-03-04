"""
plot_final_mask.py
==================

Goal
----
Quick QC visualization of the FINAL instance masks produced by the pipeline:

- mask_instances_minEqDiam8px.tif                (filtered only by minimum equivalent diameter)
- mask_instances_minEqDiam8px_tau0-12.tif        (filtered by diameter + tau_phase range)

This script makes a single side-by-side PNG showing both instance masks as a
colored label image (label2rgb), so you can quickly verify:
- if objects are being removed as expected by the tau filter,
- if instance labeling looks reasonable,
- if the mask alignment / mosaic assembly is correct.

How it works
------------
1) You provide ONE TIFF path (TIFF_PATH) inside a mask_* folder. For convenience,
   it can be either the diam-only mask or the tau-filtered mask.
2) The script infers the mask folder as TIFF_PATH.parent.
3) It loads BOTH masks from that folder by fixed names:
      mask_instances_minEqDiam8px.tif
      mask_instances_minEqDiam8px_tau0-12.tif
4) It renders each as an RGB label image (background black) and saves:
      QC_instances_diam_vs_tau.png
   inside the same mask_* folder.

Inputs
------
- TIFF_PATH: any TIFF that lives inside the mask_* folder you want to QC
  (recommended: the tau-filtered instance mask).

Outputs
-------
Inside the same mask_* folder:
- QC_instances_diam_vs_tau.png

Notes
-----
- The instance masks are expected to be uint32 label images:
    0 = background, 1..N = object labels
- Colors are arbitrary (random-ish mapping per label). This is just for QC.

How to run
----------
Edit TIFF_PATH at the top and run:
    python plot_final_mask.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from skimage.color import label2rgb


# ============================================================
# CONFIG  👉 EDIT ONLY THIS
# Provide a DIRECT TIFF path inside the target mask_* folder
# ============================================================
TIFF_PATH = Path(
    "/Users/schutyb/Documents/balu_lab/data_patient_449/" \
    "visit_04/Mosaic07_4x4_FOV600_z150_32Sp/" \
    "mask_Mosaic07_4x4_FOV600_z150_32Sp/" \
    "mask_instances_features_minEqDiam8px_tau0-12.csv" 
)
OUT_PNG_NAME = "QC_instances_diam_vs_tau.png"


# ============================================================
# Helpers
# ============================================================
def load_u32(path: Path) -> np.ndarray:
    """Load a uint32 instance mask TIFF (0=bg, 1..N=instances)."""
    return tiff.imread(str(path)).astype(np.uint32)


def plot_instances(inst_diam: np.ndarray, inst_tau: np.ndarray, out_png: Path, title: str):
    """
    Render side-by-side label2rgb views of diam-only vs diam+tau instance masks.
    Background is black.
    """
    rgb_d = label2rgb(inst_diam, bg_label=0, bg_color=(0, 0, 0))
    rgb_t = label2rgb(inst_tau,  bg_label=0, bg_color=(0, 0, 0))

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=600)

    axes[0].imshow(rgb_d)
    axes[0].set_title(f"Instances (minEqDiam ≥ 8 px)\nN = {int(inst_diam.max())}")
    axes[0].axis("off")

    axes[1].imshow(rgb_t)
    axes[1].set_title(f"Instances (minEqDiam ≥ 8 px + τphase ∈ [0,12] ns)\nN = {int(inst_tau.max())}")
    axes[1].axis("off")

    fig.suptitle(title, y=0.98)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    if not TIFF_PATH.exists():
        raise FileNotFoundError(f"TIFF_PATH does not exist:\n{TIFF_PATH}")

    # You pass a TIFF inside the mask folder -> infer the mask dir
    mask_dir = TIFF_PATH.parent

    # We always load both "diam-only" and "diam+tau" masks from that folder
    p_diam = mask_dir / "mask_instances_minEqDiam8px.tif"
    p_tau  = mask_dir / "mask_instances_minEqDiam8px_tau0-12.tif"

    if not p_diam.exists():
        raise FileNotFoundError(f"Missing: {p_diam}")
    if not p_tau.exists():
        raise FileNotFoundError(f"Missing: {p_tau}")

    inst_diam = load_u32(p_diam)
    inst_tau  = load_u32(p_tau)

    out_png = mask_dir / OUT_PNG_NAME

    # Use mosaic folder name as title (parent of mask_dir)
    title = mask_dir.parent.name

    plot_instances(inst_diam, inst_tau, out_png, title)

    print(f"[SAVED] {out_png}")
    print(f"  Mask dir: {mask_dir}")
    print(f"  Instances (diam only): {int(inst_diam.max())}")
    print(f"  Instances (diam + tau): {int(inst_tau.max())}")


if __name__ == "__main__":
    main()