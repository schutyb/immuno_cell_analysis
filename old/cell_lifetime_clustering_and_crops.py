#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
import imageio.v3 as iio
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from skimage.measure import label
from scipy.ndimage import binary_fill_holes, gaussian_filter

from phasorpy.plot import plot_phasor

from color_scales import phase_to_rgb, normalize_percentile


# ============================================================
# CONFIG
# ============================================================

PATIENT_DIR = Path("/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449")
ANALYSIS_DIR = PATIENT_DIR / "analysis"

ROI_LABELS_CSV = ANALYSIS_DIR / "roi_phasor_points_with_gmm_labels_all_three_types.csv"

PHASOR_TYPE = "uncalibrated_elastin_corr"
BIO_LABEL = "cells"

N_CLUSTERS = 3
FREQ_MHZ = 80.0

OUTPUT_DIR = ANALYSIS_DIR / "cell_families_lifetime"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CLUSTER_CSV = OUTPUT_DIR / "cell_family_clusters.csv"
OUT_CLUSTER_SUMMARY_CSV = OUTPUT_DIR / "cell_family_cluster_summary.csv"
OUT_PHASOR_PNG = OUTPUT_DIR / "cell_family_phasor_clusters_four_visits.png"

MAX_CROPS_PER_CLUSTER_PER_VISIT = 30
CROP_SIZE = 45
PADDING = 2

SHOW_PLOTS = True

CLUSTER_COLORS = {
    0: "blue",
    1: "orange",
    2: "red",
}

VISIT_ORDER = ["visit01", "visit02", "visit03", "visit04"]

# ------------------------------------------------------------
# ROI-LEVEL GLOBAL COLOR NORMALIZATION
# one color per ROI, same scale for all visits/families
# ------------------------------------------------------------
PHASE_SCALE = "reds_to_greens"
PHASE_GAMMA = 0.6

# if None, compute from ROI mean phasors of all cells
PHASE_MIN_DEG = None
PHASE_MAX_DEG = None

PHASE_PERCENTILE_MIN = 5.0
PHASE_PERCENTILE_MAX = 95.0

# ------------------------------------------------------------
# DISPLAY / BLENDING
# ------------------------------------------------------------
# final_rgb = COLOR_WEIGHT * roi_color + INTENSITY_WEIGHT * grayscale_intensity
COLOR_WEIGHT = 0.60
INTENSITY_WEIGHT = 0.40

INTENSITY_PMIN = 1.0
INTENSITY_PMAX = 99.0
INTENSITY_GAMMA = 0.95

# fill holes and soften borders for prettier display
FILL_HOLES = True
MASK_SOFTEN_SIGMA = 0.8


# ============================================================
# HELPERS
# ============================================================

def read_mask(mask_path: Path) -> np.ndarray:
    if mask_path.suffix.lower() in {".tif", ".tiff"}:
        mask = tifffile.imread(mask_path)
    else:
        mask = iio.imread(mask_path)

    mask = np.asarray(mask).squeeze()

    if mask.ndim == 3:
        if mask.shape[-1] in (3, 4):
            if np.all(mask[..., 0] == mask[..., 1]):
                mask = mask[..., 0]
            else:
                mask = mask[..., :3].max(axis=-1)
        else:
            raise ValueError(f"Unsupported mask shape: {mask.shape}")

    if mask.dtype == bool:
        return label(mask).astype(np.int32)

    unique_vals = np.unique(mask)
    if set(unique_vals.tolist()).issubset({0, 1, 255}):
        return label(mask > 0).astype(np.int32)

    return mask.astype(np.int32)


def short_visit_label(visit: str) -> str:
    visit = str(visit).lower()
    if visit.startswith("visit"):
        try:
            return f"Visit {int(visit.replace('visit', '')):02d}"
        except ValueError:
            return visit
    return visit


def phasor_to_lifetime_ns(g, s, freq_mhz):
    omega = 2.0 * np.pi * freq_mhz * 1e6
    phi = np.arctan2(s, g)
    mod = np.sqrt(g**2 + s**2)

    tau_phi = np.tan(phi) / omega
    inside = np.maximum(1.0 / np.maximum(mod**2, 1e-12) - 1.0, 0.0)
    tau_m = np.sqrt(inside) / omega

    return tau_phi * 1e9, tau_m * 1e9


def read_phasor_stack(phasor_path: Path):
    arr = tifffile.imread(phasor_path)
    arr = np.asarray(arr).squeeze()
    if arr.ndim != 3 or arr.shape[0] < 3:
        raise ValueError(f"Unexpected phasor stack shape: {arr.shape} in {phasor_path}")
    intensity = arr[0].astype(np.float64)
    g = arr[1].astype(np.float64)
    s = arr[2].astype(np.float64)
    return intensity, g, s


def find_mask_in_new_folder(mosaic_dir: Path) -> Path | None:
    candidates = [
        mosaic_dir / "_new" / "instance_mask_filtered.tif",
        mosaic_dir / "_new" / "instance_mask_filtered.tiff",
        mosaic_dir / "_new" / "instance_mask_filtered.png",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def extract_square_window(arr, rmin, rmax, cmin, cmax, crop_size, padding):
    r0 = max(rmin - padding, 0)
    r1 = min(rmax + padding, arr.shape[0])
    c0 = max(cmin - padding, 0)
    c1 = min(cmax + padding, arr.shape[1])

    crop = arr[r0:r1, c0:c1]

    h, w = crop.shape
    out = np.zeros((crop_size, crop_size), dtype=crop.dtype)

    hh = min(h, crop_size)
    ww = min(w, crop_size)

    src_r0 = max((h - hh) // 2, 0)
    src_c0 = max((w - ww) // 2, 0)

    dst_r0 = max((crop_size - hh) // 2, 0)
    dst_c0 = max((crop_size - ww) // 2, 0)

    out[dst_r0:dst_r0+hh, dst_c0:dst_c0+ww] = crop[src_r0:src_r0+hh, src_c0:src_c0+ww]
    return out


def save_phase_legend_txt(path: Path, phase_min_deg: float, phase_max_deg: float):
    txt = [
        f"scale = {PHASE_SCALE}",
        f"phase_min_deg = {phase_min_deg:.6f}",
        f"phase_max_deg = {phase_max_deg:.6f}",
        f"phase_gamma = {PHASE_GAMMA}",
        f"phase_percentile_min = {PHASE_PERCENTILE_MIN}",
        f"phase_percentile_max = {PHASE_PERCENTILE_MAX}",
        "coloring_mode = roi_uniform_color + grayscale_intensity_blend",
        f"color_weight = {COLOR_WEIGHT}",
        f"intensity_weight = {INTENSITY_WEIGHT}",
        f"intensity_pmin = {INTENSITY_PMIN}",
        f"intensity_pmax = {INTENSITY_PMAX}",
        f"intensity_gamma = {INTENSITY_GAMMA}",
        f"fill_holes = {FILL_HOLES}",
        f"mask_soften_sigma = {MASK_SOFTEN_SIGMA}",
    ]
    path.write_text("\n".join(txt))


def make_soft_roi_mask(roi_crop: np.ndarray) -> np.ndarray:
    mask = roi_crop.astype(bool)

    if FILL_HOLES:
        mask = binary_fill_holes(mask)

    soft = gaussian_filter(mask.astype(np.float32), sigma=MASK_SOFTEN_SIGMA)
    if np.max(soft) > 0:
        soft = soft / np.max(soft)
    return np.clip(soft, 0.0, 1.0)


def roi_uniform_rgb_with_intensity(
    roi_crop: np.ndarray,
    intensity_crop: np.ndarray,
    g_mean: float,
    s_mean: float,
    phase_min_deg: float,
    phase_max_deg: float,
    scale: str,
    phase_gamma: float,
) -> np.ndarray:
    # single color for the whole ROI from ROI mean phase
    phase_deg = float(np.degrees(np.arctan2(s_mean, g_mean)))

    rgb_color = phase_to_rgb(
        phase_deg=np.array([[phase_deg]], dtype=np.float32),
        scale=scale,
        phase_min_deg=phase_min_deg,
        phase_max_deg=phase_max_deg,
        phase_gamma=phase_gamma,
    )[0, 0]

    # grayscale from real intensity texture
    inten_norm = normalize_percentile(
        intensity_crop,
        pmin=INTENSITY_PMIN,
        pmax=INTENSITY_PMAX,
    )
    if INTENSITY_GAMMA != 1.0:
        inten_norm = inten_norm ** INTENSITY_GAMMA

    gray_rgb = np.stack([inten_norm, inten_norm, inten_norm], axis=-1)

    # soft ROI alpha mask
    alpha = make_soft_roi_mask(roi_crop)[..., None]

    # blend color + grayscale inside ROI
    roi_rgb = COLOR_WEIGHT * rgb_color[None, None, :] + INTENSITY_WEIGHT * gray_rgb
    roi_rgb = np.clip(roi_rgb, 0.0, 1.0)

    # black background outside ROI
    out = roi_rgb * alpha
    return np.clip(out, 0.0, 1.0)


# ============================================================
# MAIN
# ============================================================

def main():
    if not ROI_LABELS_CSV.exists():
        raise FileNotFoundError(f"Missing ROI labels CSV: {ROI_LABELS_CSV}")

    df = pd.read_csv(ROI_LABELS_CSV)
    required = {"visit", "mosaic_name", "phasor_type", "bio_label", "roi_label", "g_mean", "s_mean", "phasor_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in ROI labels CSV: {sorted(missing)}")

    df["visit"] = df["visit"].astype(str).str.lower()
    df["bio_label"] = df["bio_label"].astype(str).str.lower()

    df = df[
        (df["phasor_type"] == PHASOR_TYPE) &
        (df["bio_label"] == BIO_LABEL)
    ].copy()

    if df.empty:
        raise RuntimeError("No cell ROIs found for selected phasor type.")

    tau_phi, tau_m = phasor_to_lifetime_ns(df["g_mean"].to_numpy(), df["s_mean"].to_numpy(), FREQ_MHZ)
    df["tau_phi_ns"] = tau_phi
    df["tau_m_ns"] = tau_m
    df["phase_rad"] = np.arctan2(df["s_mean"], df["g_mean"])
    df["phase_deg"] = np.degrees(df["phase_rad"])
    df["modulation"] = np.sqrt(df["g_mean"]**2 + df["s_mean"]**2)

    # --------------------------------------------------------
    # 3-cluster GMM on cell phasor coordinates
    # --------------------------------------------------------
    X = df[["g_mean", "s_mean"]].to_numpy()
    gmm = GaussianMixture(
        n_components=N_CLUSTERS,
        covariance_type="full",
        random_state=0,
        n_init=10,
    )
    raw_cluster = gmm.fit_predict(X)
    df["cluster_raw"] = raw_cluster

    # reorder clusters by mean phase lifetime
    order = (
        df.groupby("cluster_raw")["tau_phi_ns"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    remap = {old: new for new, old in enumerate(order)}
    df["cell_family"] = df["cluster_raw"].map(remap)

    summary = (
        df.groupby(["visit", "cell_family"])
        .agg(
            n_cells=("roi_label", "size"),
            g_mean=("g_mean", "mean"),
            s_mean=("s_mean", "mean"),
            phase_deg_mean=("phase_deg", "mean"),
            phase_deg_median=("phase_deg", "median"),
            tau_phi_mean_ns=("tau_phi_ns", "mean"),
            tau_phi_median_ns=("tau_phi_ns", "median"),
            tau_m_mean_ns=("tau_m_ns", "mean"),
            tau_m_median_ns=("tau_m_ns", "median"),
        )
        .reset_index()
        .sort_values(["visit", "cell_family"])
    )

    df.to_csv(OUT_CLUSTER_CSV, index=False)
    summary.to_csv(OUT_CLUSTER_SUMMARY_CSV, index=False)

    print("[INFO] Cluster summary by visit:")
    print(summary)

    # ========================================================
    # 1) PLOT PHASOR CLUSTERIZED - 4 PANELS
    # ========================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, visit in zip(axes, VISIT_ORDER):
        dv = df[df["visit"] == visit].copy()

        if dv.empty:
            plot_phasor(
                np.array([0.5]),
                np.array([0.0]),
                style="plot",
                marker="",
                linestyle="",
                frequency=FREQ_MHZ,
                ax=ax,
                title=short_visit_label(visit),
                show=False,
            )
            ax.text(0.5, 0.35, "No cell ROIs", ha="center", va="center", fontsize=12)
            continue

        plotted_any = False
        for fam in sorted(dv["cell_family"].unique()):
            dvc = dv[dv["cell_family"] == fam]

            plot_phasor(
                dvc["g_mean"].to_numpy(),
                dvc["s_mean"].to_numpy(),
                style="plot",
                marker=".",
                linestyle="",
                color=CLUSTER_COLORS[int(fam)],
                label=f"Family {fam} (n={len(dvc)})",
                frequency=FREQ_MHZ,
                ax=ax,
                title=short_visit_label(visit),
                show=False,
            )
            plotted_any = True

        if not plotted_any:
            plot_phasor(
                np.array([0.5]),
                np.array([0.0]),
                style="plot",
                marker="",
                linestyle="",
                frequency=FREQ_MHZ,
                ax=ax,
                title=short_visit_label(visit),
                show=False,
            )

        for fam in sorted(dv["cell_family"].unique()):
            dvc = dv[dv["cell_family"] == fam]
            ax.scatter(
                dvc["g_mean"].mean(),
                dvc["s_mean"].mean(),
                s=140,
                c=CLUSTER_COLORS[int(fam)],
                edgecolors="black",
                linewidths=1.0,
                marker="X",
                zorder=10,
            )

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 0.7)
        ax.legend(fontsize=8)

    fig.suptitle("Cell families in phasor space (3-cluster GMM)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_PHASOR_PNG, dpi=300, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

    # ========================================================
    # 2) PICK REPRESENTATIVE ROIS FOR CROPS
    # ========================================================
    rep_rows = []

    for visit in sorted(df["visit"].unique()):
        dv = df[df["visit"] == visit].copy()

        for fam in sorted(dv["cell_family"].unique()):
            dvc = dv[dv["cell_family"] == fam].copy()
            if dvc.empty:
                continue

            cg = dvc["g_mean"].mean()
            cs = dvc["s_mean"].mean()
            dvc["dist_to_centroid"] = np.sqrt((dvc["g_mean"] - cg) ** 2 + (dvc["s_mean"] - cs) ** 2)
            dvc = dvc.sort_values("dist_to_centroid").head(MAX_CROPS_PER_CLUSTER_PER_VISIT)

            rep_rows.append(dvc)

    if rep_rows:
        df_reps = pd.concat(rep_rows, ignore_index=True)
    else:
        df_reps = pd.DataFrame(columns=df.columns)

    # ========================================================
    # 3) GLOBAL ROI-LEVEL PHASE RANGE
    # ========================================================
    if PHASE_MIN_DEG is None or PHASE_MAX_DEG is None:
        all_phase_deg = df["phase_deg"].to_numpy(dtype=float)
        all_phase_deg = all_phase_deg[np.isfinite(all_phase_deg)]
        if all_phase_deg.size == 0:
            raise RuntimeError("Could not compute global ROI phase range.")
        phase_min_deg = float(np.percentile(all_phase_deg, PHASE_PERCENTILE_MIN))
        phase_max_deg = float(np.percentile(all_phase_deg, PHASE_PERCENTILE_MAX))
    else:
        phase_min_deg = float(PHASE_MIN_DEG)
        phase_max_deg = float(PHASE_MAX_DEG)

    print(f"[INFO] Global ROI phase range (deg): {phase_min_deg:.3f} -> {phase_max_deg:.3f}")
    save_phase_legend_txt(OUTPUT_DIR / "crop_colormap_settings.txt", phase_min_deg, phase_max_deg)

    # ========================================================
    # 4) EXTRACT GLOBAL-NORMALIZED ROI-UNIFORM CROPS
    # ========================================================
    saved = 0
    skipped = 0

    for _, row in df_reps.iterrows():
        phasor_path = Path(row["phasor_path"])
        mosaic_dir = phasor_path.parent

        mask_path = find_mask_in_new_folder(mosaic_dir)
        if mask_path is None:
            skipped += 1
            continue

        try:
            intensity_img, _, _ = read_phasor_stack(phasor_path)
            labels = read_mask(mask_path)

            roi_label = int(row["roi_label"])
            roi_mask = labels == roi_label
            if not np.any(roi_mask):
                skipped += 1
                continue

            rr, cc = np.where(roi_mask)
            rmin, rmax = rr.min(), rr.max() + 1
            cmin, cmax = cc.min(), cc.max() + 1

            intensity_crop = extract_square_window(
                intensity_img, rmin, rmax, cmin, cmax, CROP_SIZE, PADDING
            )

            full_mask = roi_mask.astype(np.uint8)
            roi_crop = extract_square_window(
                full_mask, rmin, rmax, cmin, cmax, CROP_SIZE, PADDING
            ) > 0

            rgb_float = roi_uniform_rgb_with_intensity(
                roi_crop=roi_crop,
                intensity_crop=intensity_crop,
                g_mean=float(row["g_mean"]),
                s_mean=float(row["s_mean"]),
                phase_min_deg=phase_min_deg,
                phase_max_deg=phase_max_deg,
                scale=PHASE_SCALE,
                phase_gamma=PHASE_GAMMA,
            )

            rgb = np.clip(rgb_float * 255.0, 0, 255).astype(np.uint8)

            visit = str(row["visit"]).lower()
            fam = int(row["cell_family"])

            out_dir = OUTPUT_DIR / "crops" / visit / f"family_{fam}"
            out_dir.mkdir(parents=True, exist_ok=True)

            out_name = f"{row['mosaic_name']}_roi{int(row['roi_label']):04d}.png"
            out_path = out_dir / out_name
            iio.imwrite(out_path, rgb)
            saved += 1

        except Exception as e:
            print(f"[WARN] Failed crop for {row['mosaic_name']} roi={row['roi_label']}: {e}")
            skipped += 1

    print(f"[DONE] Saved clustered ROI table: {OUT_CLUSTER_CSV}")
    print(f"[DONE] Saved cluster summary: {OUT_CLUSTER_SUMMARY_CSV}")
    print(f"[DONE] Saved phasor plot: {OUT_PHASOR_PNG}")
    print(f"[DONE] ROI-uniform blended crops saved: {saved}")
    print(f"[DONE] ROI-uniform blended crops skipped: {skipped}")
    print(f"[DONE] Output dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()