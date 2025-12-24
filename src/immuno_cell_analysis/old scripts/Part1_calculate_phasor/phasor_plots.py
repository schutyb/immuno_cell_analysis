"""
phasor_plots.py
===============

Goal
----
Quality-control (QC) + visualization for calibrated phasor mosaics.

This script:
1) Finds every `phasor_calibrated_CYX.tif` under a phasor output folder (recursive).
2) Loads each calibrated CYX phasor TIFF.
3) Applies a median filter in phasor space (mean, real, imag) using phasorpy:
      phasor_filter_median(mean, real, imag, size=..., repeat=...)
4) Saves:
   - a filtered CYX TIFF: `phasor_filtered_median_CYX.tif`
   - QC images (JPG):
       * mean_image.jpg        (grayscale mean intensity, robust display range)
       * mean_histogram.jpg    (histogram of mean intensity)
       * phasor_H1.jpg         (PhasorPlot hist2d for H1 with white background for low bins)

Why median filtering?
--------------------
The raw per-pixel phasor clouds can look very spread/noisy due to low photon counts,
shot noise, and pixel-level variability. Median filtering in (g,s) (via real/imag)
reduces speckle and stabilizes the phasor representation for visualization/QC.

Input files
-----------
Requires calibrated phasor TIFFs produced by `calculate_phasor.py`:

    phasor_calibrated_CYX.tif

Expected axes:
    (C, Y, X)  float32
with channels:

    C0: mean
    C1: g1  (real H1)
    C2: s1  (imag H1)
    C3: mod1
    C4: phase1
    C5: g2  (real H2)
    C6: s2  (imag H2)
    C7: mod2
    C8: phase2

Outputs
-------
For each calibrated TIFF found at:
    <PHASOR_ROOT>/<...>/phasor_calibrated_CYX.tif

This script writes to:
    <OUT_ROOT>/<...>/

Files created:
    phasor_filtered_median_CYX.tif
    mean_image.jpg
    mean_histogram.jpg
    phasor_H1.jpg

Visualization details (important)
---------------------------------
PhasorPlot uses a histogram (2D bins). Some bins have very small counts and make the
background look “dirty”/colored instead of white.

To fix that, we compute a per-plot threshold `cmin` based on a fraction of the max bin:
    cmin = ceil(MIN_FRAC_OF_MAX * max_bin_count)

Then:
- bins with counts < cmin are shown as "under" the colormap -> rendered white
- background becomes clean white while keeping main phasor distribution

How to run
----------
1) Edit PHASOR_ROOT and OUT_ROOT in main()
2) Run:
      python phasor_plots.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

from phasorpy.phasor import phasor_to_polar
from phasorpy.filter import phasor_filter_median
from phasorpy.plot import PhasorPlot


# ============================================================
# CONFIG (edit these parameters)
# ============================================================
# Median filter parameters (phasor domain)
FILTER_SIZE = 7
FILTER_REPEAT = 2

# Display-only visualization cutoff:
# hide histogram bins below MIN_FRAC_OF_MAX * max_bin_count (render as white background).
MIN_FRAC_OF_MAX = 0.002  # 0.2% of the max bin count

# JPG output quality
JPG_DPI = 200

# PhasorPlot rendering params
PHASOR_BINS = 512
PHASOR_CMAP = "RdYlGn_r"
PHASOR_XLIM = (-0.05, 1.05)
PHASOR_YLIM = (-0.05, 0.70)


# ============================================================
# IO helpers
# ============================================================
def load_cyx_tiff(path: Path) -> np.ndarray:
    """Load a CYX TIFF as float32 (C,Y,X)."""
    arr = tiff.imread(str(path))
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected (C,Y,X) TIFF. Got {arr.shape} for {path}")
    return arr.astype(np.float32)


def save_tiff_cyx(path: Path, cyx: np.ndarray):
    """Save a float32 CYX TIFF with axes metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(
        str(path),
        cyx.astype(np.float32),
        photometric="minisblack",
        metadata={"axes": "CYX"},
    )


# ============================================================
# QC plots
# ============================================================
def save_mean_image_jpg(out_path: Path, mean_img: np.ndarray):
    """
    Save grayscale mean intensity image using robust display range to avoid saturation.
    Display range: vmin/vmax = p1 and p99.8 percentiles (finite pixels only)
    """
    v = mean_img[np.isfinite(mean_img)]
    if v.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = np.percentile(v, [1.0, 99.8])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin = float(np.nanmin(v))
            vmax = float(np.nanmax(v))
            if vmax <= vmin:
                vmax = vmin + 1e-6

    plt.figure(figsize=(7, 6))
    plt.imshow(
        mean_img,
        cmap="gray",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    plt.title("Mean intensity (grayscale)\n(display p1–p99.8)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=JPG_DPI, facecolor="white")
    plt.close()


def save_mean_histogram_gray_jpg(out_path: Path, mean_img: np.ndarray):
    """Save histogram of mean intensity (finite pixels only)."""
    v = mean_img[np.isfinite(mean_img)].ravel()
    plt.figure(figsize=(7, 5))
    plt.hist(v, bins=256, color="gray")
    plt.title("Histogram of mean intensity (gray)")
    plt.xlabel("Mean intensity")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=JPG_DPI, facecolor="white")
    plt.close()


def save_phasor_h1_phasorplot_relative_jpg(
    out_path: Path,
    g_h1: np.ndarray,
    s_h1: np.ndarray,
    title: str,
):
    """
    Save H1 phasor histogram using PhasorPlot.hist2d with a clean white background.
    We compute cmin from the histogram's max bin count:
        cmin = ceil(MIN_FRAC_OF_MAX * max_count), at least 1
    Then bins below vmin=cmin become "under" -> rendered white.
    """
    m = np.isfinite(g_h1) & np.isfinite(s_h1)
    g = g_h1[m].ravel()
    s = s_h1[m].ravel()

    # Compute histogram first to determine max_count -> cmin
    H, _, _ = np.histogram2d(
        g,
        s,
        bins=PHASOR_BINS,
        range=[PHASOR_XLIM, PHASOR_YLIM],
    )
    max_count = float(H.max()) if H.size else 0.0
    cmin = int(max(1, np.ceil(MIN_FRAC_OF_MAX * max_count))) if max_count > 0 else 1

    cmap = plt.get_cmap(PHASOR_CMAP).copy()
    cmap.set_under("white")  # bins below vmin are white

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    ax.set_facecolor("white")

    pp = PhasorPlot(ax=ax)
    pp.hist2d(
        g,
        s,
        cmap=cmap,
        bins=PHASOR_BINS,
        cmin=cmin,
        vmin=cmin,
    )

    ax.set_xlim(*PHASOR_XLIM)
    ax.set_ylim(*PHASOR_YLIM)
    ax.set_title(f"{title}\n(cmin={cmin} = {MIN_FRAC_OF_MAX*100:.2f}% of max bin)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=JPG_DPI, facecolor="white")
    plt.close(fig)


# ============================================================
# Filtering on calibrated CYX TIFF
# ============================================================
def filter_calibrated_cyx(cyx: np.ndarray) -> np.ndarray:
    """
    Apply phasor median filtering to a calibrated CYX TIFF.
    Filtering is applied to mean + real/imag for H1/H2, then phase/mod recomputed.
    """
    mean = cyx[0]  # (Y,X)

    # Build (H,Y,X) arrays for H1 and H2
    real = np.stack([cyx[1], cyx[5]], axis=0)
    imag = np.stack([cyx[2], cyx[6]], axis=0)

    mean_f, real_f, imag_f = phasor_filter_median(
        mean,
        real,
        imag,
        size=int(FILTER_SIZE),
        repeat=int(FILTER_REPEAT),
    )

    phase_f, mod_f = phasor_to_polar(real_f, imag_f)

    out = np.stack(
        [
            mean_f,
            real_f[0], imag_f[0], mod_f[0], phase_f[0],
            real_f[1], imag_f[1], mod_f[1], phase_f[1],
        ],
        axis=0,
    ).astype(np.float32)

    return out


def find_calibrated_tiffs(root: Path) -> list[Path]:
    """Recursively find all calibrated phasor mosaics."""
    return sorted(root.rglob("phasor_calibrated_CYX.tif"))


# ============================================================
# Main worker
# ============================================================
def run_qc(phasor_root: Path, out_root: Path):
    """
    Run QC for all calibrated phasor TIFFs under `phasor_root`.
    Creates a mirrored folder structure under `out_root`.
    """
    cal_paths = find_calibrated_tiffs(phasor_root)
    if not cal_paths:
        raise FileNotFoundError(f"No phasor_calibrated_CYX.tif found under: {phasor_root}")

    print(f"[INFO] Found {len(cal_paths)} calibrated TIFF(s).")

    for cal_path in cal_paths:
        rel = cal_path.relative_to(phasor_root)
        out_dir = out_root / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[CAL] {cal_path}")

        cyx_cal = load_cyx_tiff(cal_path)
        cyx_filt = filter_calibrated_cyx(cyx_cal)

        # Save filtered phasor TIFF
        save_tiff_cyx(out_dir / "phasor_filtered_median_CYX.tif", cyx_filt)

        # QC plots
        save_mean_image_jpg(out_dir / "mean_image.jpg", cyx_cal[0])
        save_mean_histogram_gray_jpg(out_dir / "mean_histogram.jpg", cyx_cal[0])
        save_phasor_h1_phasorplot_relative_jpg(
            out_dir / "phasor_H1.jpg",
            cyx_filt[1],  # g1 filtered
            cyx_filt[2],  # s1 filtered
            title=f"Phasor H1 (median {FILTER_SIZE}x{FILTER_SIZE}, repeat={FILTER_REPEAT})",
        )

        print("  [SAVED]", out_dir)


# ============================================================
# Entry point (for preflight + run_all)
# ============================================================
def main():
    # Input: output folder from calculate_phasor.py
    PHASOR_ROOT = Path("/Users/schutyb/Documents/balu_lab/data_patient_449/phasor_out")

    # Output: filtered TIFFs + QC plots
    OUT_ROOT = Path("/Users/schutyb/Documents/balu_lab/data_patient_449/phasor_qc")

    run_qc(PHASOR_ROOT, OUT_ROOT)


if __name__ == "__main__":
    main()