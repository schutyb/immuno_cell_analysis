#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build final cell masks from candidate segmentation masks using area filtering
and ROI-level phasor/GMM classification.

This script processes all visits and mosaics inside a patient folder. For each
mosaic, it reads:

    1. candidate segmentation masks from SegData/
    2. the initial calibrated phasor mosaic from phasor/

The candidate masks are first filtered by object area. Then, for each remaining
ROI, the script computes ROI-mean calibrated phasor coordinates using the
selected detector channel, usually the green detector.

A Gaussian Mixture Model is then fitted in ROI phasor space to separate tissue
classes.

Expected biological classes:
    - cell and elastin are always expected
    - melanin is optional and depends on the visit/patient

Class assignment is based on mean phase ordering:

For 2 components:
    lowest phase  -> cell
    highest phase -> elastin

For 3 components:
    lowest phase       -> melanin
    intermediate phase -> cell
    highest phase      -> elastin

Final cell ROIs are kept only if:
    lifetime_class == "cell"
    and cell_probability >= CELL_PROB_THRESHOLD

Outputs:
    - final cell mask for the full mosaic
    - lifetime class mask for the full mosaic
    - optional debug masks for melanin, elastin, and low-confidence cells
    - final masks split back into individual tiles
    - ROI-level CSV with area, phasor, GMM class, and probabilities
    - GMM cluster summary CSV
    - elastin center-of-mass CSV
    - GMM phasor QC plot

Important:
    - This script uses the initial coumarin-calibrated phasor.
    - It does not perform elastin-based correction.
    - The elastin summary produced here can be used later for correction.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tiff
from PIL import Image
from skimage.measure import label, regionprops
from skimage.transform import resize
from sklearn.mixture import GaussianMixture

# =========================
# CONFIG
# =========================

PATIENT_DIR = Path(
    "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449"
).expanduser()

PHASOR_SUBDIR = "phasor"
PHASOR_NAME = "phasor_calibrated_green_blue_mosaic.tif"

SEG_SUBDIR = "SegData"

OUTPUT_SUBDIR = "final_masks_area_phasor_gmm"

MIN_AREA_PX = 20
CELL_PROB_THRESHOLD = 0.5

# Phasor planes:
# 0 = DC
# 1 = G green
# 2 = S green
# 3 = G blue
# 4 = S blue
G_PLANE = 1
S_PLANE = 2

# =========================
# GMM CONFIGURATION
# =========================
#
# Define the expected number of phasor components for each visit.
#
# 2 components:
#     cell + elastin
#
# 3 components:
#     melanin + cell + elastin
#
# This is patient/visit-dependent. Edit this dictionary before running the
# script for a new patient.
#
# If a visit is not listed, DEFAULT_N_COMPONENTS is used.

VISIT_GMM_COMPONENTS = {
    "visit01": 3,
    "visit02": 3,
    "visit03": 3,
    "visit04": 2,
}

DEFAULT_N_COMPONENTS = 3

RANDOM_STATE = 0
SAVE_DEBUG_CLASS_MASKS = True


# =========================
# HELPERS
# =========================


def natural_key(path):
    return [
        int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(path.name))
    ]


def parse_mosaic_shape_from_name(folder_name):
    match = re.search(r"(\d+)x(\d+)", folder_name)

    if not match:
        raise ValueError(f"No pude detectar forma del mosaico en: {folder_name}")

    return int(match.group(1)), int(match.group(2))


def extract_tile_number(name):
    patterns = [
        r"Im_(\d+)",
        r"_t(\d+)",
        r"tile[_-]?(\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, name, re.IGNORECASE)

        if match:
            return int(match.group(1))

    numbers = re.findall(r"\d+", name)

    return int(numbers[-1]) if numbers else None


def collect_mask_paths(seg_dir):
    files = []

    for ext in ("*.tif", "*.tiff", "*.png"):
        files.extend(seg_dir.glob(ext))

    pairs = []

    for path in files:
        tile_number = extract_tile_number(path.name)

        if tile_number is not None:
            pairs.append((tile_number, path))

    pairs.sort(key=lambda item: item[0])

    return [path for _, path in pairs]


def read_mask(path):
    if path.suffix.lower() in [".tif", ".tiff"]:
        arr = tiff.imread(path)
    else:
        arr = np.array(Image.open(path))

    if arr.ndim == 3:
        arr = arr[..., 0]

    return arr > 0


def snake_indices(nrows, ncols):
    indices = np.arange(nrows * ncols).reshape(nrows, ncols)
    layout = []

    for row in range(nrows):
        row_indices = indices[row].copy()

        if row % 2 == 1:
            row_indices = row_indices[::-1]

        layout.append(row_indices.tolist())

    return layout


def stitch_tiles_snake(tile_images, nrows, ncols):
    expected = nrows * ncols

    if len(tile_images) != expected:
        raise ValueError(f"Esperaba {expected} tiles, recibí {len(tile_images)}")

    height, width = tile_images[0].shape

    mosaic = np.zeros(
        (nrows * height, ncols * width),
        dtype=tile_images[0].dtype,
    )

    layout = snake_indices(nrows, ncols)

    for row in range(nrows):
        for col in range(ncols):
            tile_index = layout[row][col]

            y0 = row * height
            y1 = y0 + height
            x0 = col * width
            x1 = x0 + width

            mosaic[y0:y1, x0:x1] = tile_images[tile_index]

    return mosaic


def split_mosaic_to_tiles_snake(mosaic, nrows, ncols):
    height_total, width_total = mosaic.shape

    height = height_total // nrows
    width = width_total // ncols

    tiles = [None] * (nrows * ncols)
    layout = snake_indices(nrows, ncols)

    for row in range(nrows):
        for col in range(ncols):
            tile_index = layout[row][col]

            y0 = row * height
            y1 = y0 + height
            x0 = col * width
            x1 = x0 + width

            tiles[tile_index] = mosaic[y0:y1, x0:x1]

    return tiles


def resize_mask_if_needed(mask, target_shape):
    if mask.shape == target_shape:
        return mask

    resized = resize(
        mask.astype(float),
        target_shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )

    return resized > 0.5


def phase_deg(g, s):
    return np.degrees(np.arctan2(s, g)).astype(np.float32)


def get_visit_name(mosaic_dir):
    return mosaic_dir.parent.name


def get_n_components_for_visit(visit_name):
    n_components = VISIT_GMM_COMPONENTS.get(visit_name, DEFAULT_N_COMPONENTS)

    if n_components not in (2, 3):
        raise ValueError(
            f"Configuración inválida para {visit_name}: "
            f"n_components={n_components}. Usar solo 2 o 3."
        )

    return n_components


def get_classes_for_n_components(n_components):
    if n_components == 2:
        return ["cell", "elastin"]

    if n_components == 3:
        return ["melanin", "cell", "elastin"]

    raise ValueError("Solo está implementado GMM con 2 o 3 componentes.")


def assign_gmm_classes(df, n_components):
    """
    Assign biological classes based on cluster mean phase ordering.

    Assumptions:
        - elastin always has the highest phase;
        - cells always have lower phase than elastin;
        - melanin, when present, has the lowest phase.

    For 2 components:
        lowest phase  -> cell
        highest phase -> elastin

    For 3 components:
        lowest phase       -> melanin
        intermediate phase -> cell
        highest phase      -> elastin
    """
    cluster_summary = (
        df.groupby("gmm_cluster")
        .agg(
            mean_phase_deg=("mean_phase_deg", "mean"),
            mean_g=("mean_g", "mean"),
            mean_s=("mean_s", "mean"),
            n_rois=("roi_id", "count"),
        )
        .reset_index()
        .sort_values("mean_phase_deg")
    )

    ordered_clusters = cluster_summary["gmm_cluster"].tolist()
    ordered_classes = get_classes_for_n_components(n_components)

    class_map = {
        cluster_id: class_name
        for cluster_id, class_name in zip(ordered_clusters, ordered_classes)
    }

    df["lifetime_class"] = df["gmm_cluster"].map(class_map)
    cluster_summary["lifetime_class"] = cluster_summary["gmm_cluster"].map(class_map)

    return df, cluster_summary


def plot_gmm_phasor(df, cluster_summary, out_path, title):
    colors = {
        "melanin": "tab:brown",
        "cell": "tab:green",
        "cell_low_confidence": "tab:gray",
        "elastin": "tab:orange",
        "unknown": "tab:gray",
    }

    plt.figure(figsize=(7, 6))

    plot_class_col = "final_class" if "final_class" in df.columns else "lifetime_class"

    for class_name, sub_df in df.groupby(plot_class_col):
        plt.scatter(
            sub_df["mean_g"],
            sub_df["mean_s"],
            s=10,
            alpha=0.65,
            label=f"{class_name} (n={len(sub_df)})",
            color=colors.get(class_name, "tab:gray"),
        )

    for _, row in cluster_summary.iterrows():
        plt.scatter(
            row["mean_g"],
            row["mean_s"],
            s=160,
            marker="x",
            linewidths=3,
            color=colors.get(row["lifetime_class"], "black"),
        )

        plt.text(
            row["mean_g"],
            row["mean_s"],
            f" {row['lifetime_class']}",
            fontsize=9,
            va="center",
        )

    theta = np.linspace(0, np.pi, 400)
    g_circle = 0.5 + 0.5 * np.cos(theta)
    s_circle = 0.5 * np.sin(theta)

    plt.plot(g_circle, s_circle, "k--", linewidth=1, alpha=0.5)

    plt.xlabel("G")
    plt.ylabel("S")
    plt.title(title)
    plt.axis("equal")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()


def make_empty_elastin_summary(visit_name, mosaic_name):
    return pd.DataFrame(
        [
            {
                "visit": visit_name,
                "mosaic": mosaic_name,
                "n_elastin_rois": 0,
                "elastin_cm_g": np.nan,
                "elastin_cm_s": np.nan,
                "elastin_cm_phase_deg": np.nan,
                "elastin_cm_modulation": np.nan,
                "elastin_std_g": np.nan,
                "elastin_std_s": np.nan,
                "elastin_std_phase_deg": np.nan,
            }
        ]
    )


def compute_elastin_summary(df, visit_name, mosaic_name):
    elastin = df[df["lifetime_class"] == "elastin"]

    if len(elastin) == 0:
        return make_empty_elastin_summary(visit_name, mosaic_name)

    return pd.DataFrame(
        [
            {
                "visit": visit_name,
                "mosaic": mosaic_name,
                "n_elastin_rois": len(elastin),
                "elastin_cm_g": float(np.nanmean(elastin["mean_g"])),
                "elastin_cm_s": float(np.nanmean(elastin["mean_s"])),
                "elastin_cm_phase_deg": float(np.nanmean(elastin["mean_phase_deg"])),
                "elastin_cm_modulation": float(np.nanmean(elastin["mean_modulation"])),
                "elastin_std_g": float(np.nanstd(elastin["mean_g"])),
                "elastin_std_s": float(np.nanstd(elastin["mean_s"])),
                "elastin_std_phase_deg": float(np.nanstd(elastin["mean_phase_deg"])),
            }
        ]
    )


# =========================
# PROCESS MOSAIC
# =========================


def process_mosaic(mosaic_dir):
    visit_name = get_visit_name(mosaic_dir)
    mosaic_name = mosaic_dir.name

    n_components = get_n_components_for_visit(visit_name)
    expected_classes = get_classes_for_n_components(n_components)

    print(f"\n[PROCESS] {visit_name} | {mosaic_name}")
    print(f"          GMM components: {n_components} -> {expected_classes}")

    phasor_path = mosaic_dir / PHASOR_SUBDIR / PHASOR_NAME
    seg_dir = mosaic_dir / SEG_SUBDIR
    out_dir = mosaic_dir / OUTPUT_SUBDIR

    out_dir.mkdir(parents=True, exist_ok=True)

    if not phasor_path.exists():
        print(f"[SKIP] No existe phasor: {phasor_path}")
        return

    if not seg_dir.exists():
        print(f"[SKIP] No existe SegData: {seg_dir}")
        return

    phasor = tiff.imread(phasor_path).astype(np.float32)

    if phasor.ndim != 3 or phasor.shape[0] < 5:
        raise ValueError(f"Phasor inválido: {phasor.shape}")

    dc = phasor[0]
    g = phasor[G_PLANE]
    s = phasor[S_PLANE]

    nrows, ncols = parse_mosaic_shape_from_name(mosaic_name)
    expected_tiles = nrows * ncols

    mask_paths = collect_mask_paths(seg_dir)

    if len(mask_paths) != expected_tiles:
        raise ValueError(
            f"{mosaic_name}: esperaba {expected_tiles} máscaras, "
            f"encontré {len(mask_paths)}"
        )

    mask_tiles = [read_mask(path) for path in mask_paths]
    mask_mosaic = stitch_tiles_snake(mask_tiles, nrows, ncols)
    mask_mosaic = resize_mask_if_needed(mask_mosaic, dc.shape)

    labeled = label(mask_mosaic, connectivity=2)

    rows = []
    keep_area_mask = np.zeros_like(mask_mosaic, dtype=bool)

    roi_id = 0

    for prop in regionprops(labeled):
        area = prop.area

        if area < MIN_AREA_PX:
            continue

        roi_mask = labeled == prop.label

        roi_g = g[roi_mask]
        roi_s = s[roi_mask]
        roi_dc = dc[roi_mask]

        mean_g = np.nanmean(roi_g)
        mean_s = np.nanmean(roi_s)
        mean_dc = np.nanmean(roi_dc)

        if not np.isfinite(mean_g) or not np.isfinite(mean_s):
            continue

        roi_id += 1

        mean_phase = float(phase_deg(mean_g, mean_s))
        mean_modulation = float(np.sqrt(mean_g**2 + mean_s**2))

        rows.append(
            {
                "visit": visit_name,
                "mosaic": mosaic_name,
                "roi_id": roi_id,
                "label_original": prop.label,
                "area_px": float(area),
                "centroid_y": float(prop.centroid[0]),
                "centroid_x": float(prop.centroid[1]),
                "mean_dc": float(mean_dc),
                "mean_g": float(mean_g),
                "mean_s": float(mean_s),
                "mean_phase_deg": mean_phase,
                "mean_modulation": mean_modulation,
            }
        )

        keep_area_mask[roi_mask] = True

    if len(rows) == 0:
        print("[WARNING] No quedaron ROIs después del filtro de área.")
        return

    df = pd.DataFrame(rows)

    if len(df) < n_components:
        print(
            f"[WARNING] ROIs insuficientes para GMM: n_rois={len(df)}, "
            f"n_components={n_components}"
        )
        return

    x = df[["mean_g", "mean_s"]].to_numpy(dtype=np.float32)

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=RANDOM_STATE,
        n_init=10,
        reg_covar=1e-6,
    )

    df["gmm_cluster"] = gmm.fit_predict(x)

    probabilities = gmm.predict_proba(x)
    df["gmm_max_probability"] = probabilities.max(axis=1)

    df, cluster_summary = assign_gmm_classes(df, n_components=n_components)

    cluster_to_class = dict(
        zip(cluster_summary["gmm_cluster"], cluster_summary["lifetime_class"])
    )

    for cluster_id, class_name in cluster_to_class.items():
        probability_column = f"prob_{class_name}"
        df[probability_column] = probabilities[:, int(cluster_id)]

    df["assigned_class_probability"] = np.nan

    for cluster_id, class_name in cluster_to_class.items():
        idx = df["gmm_cluster"] == cluster_id
        df.loc[idx, "assigned_class_probability"] = probabilities[
            idx.to_numpy(),
            int(cluster_id),
        ]

    cell_clusters = [
        cluster_id
        for cluster_id, class_name in cluster_to_class.items()
        if class_name == "cell"
    ]

    if len(cell_clusters) > 0:
        cell_cluster = int(cell_clusters[0])
        df["cell_probability"] = probabilities[:, cell_cluster]
    else:
        df["cell_probability"] = np.nan

    df["final_class"] = df["lifetime_class"]

    low_confidence_cell = (df["lifetime_class"] == "cell") & (
        df["cell_probability"] < CELL_PROB_THRESHOLD
    )

    df.loc[low_confidence_cell, "final_class"] = "cell_low_confidence"

    roi_csv = out_dir / f"{mosaic_name}_roi_area_phasor_gmm.csv"
    df.to_csv(roi_csv, index=False)

    gmm_csv = out_dir / f"{mosaic_name}_gmm_cluster_summary.csv"
    cluster_summary.to_csv(gmm_csv, index=False)

    elastin_summary = compute_elastin_summary(df, visit_name, mosaic_name)
    elastin_csv = out_dir / f"{mosaic_name}_elastin_cm.csv"
    elastin_summary.to_csv(elastin_csv, index=False)

    final_cell_mask = np.zeros_like(mask_mosaic, dtype=np.uint8)
    final_melanin_mask = np.zeros_like(mask_mosaic, dtype=np.uint8)
    final_elastin_mask = np.zeros_like(mask_mosaic, dtype=np.uint8)
    final_low_confidence_cell_mask = np.zeros_like(mask_mosaic, dtype=np.uint8)
    final_lifetime_class_mask = np.zeros_like(mask_mosaic, dtype=np.uint16)

    class_to_value = {
        "melanin": 1,
        "cell": 2,
        "elastin": 3,
        "cell_low_confidence": 4,
    }

    for _, row in df.iterrows():
        roi_mask = labeled == int(row["label_original"])
        class_name = row["final_class"]

        final_lifetime_class_mask[roi_mask] = class_to_value.get(class_name, 0)

        if class_name == "cell":
            final_cell_mask[roi_mask] = 1
        elif class_name == "melanin":
            final_melanin_mask[roi_mask] = 1
        elif class_name == "elastin":
            final_elastin_mask[roi_mask] = 1
        elif class_name == "cell_low_confidence":
            final_low_confidence_cell_mask[roi_mask] = 1

    tiff.imwrite(
        out_dir / f"{mosaic_name}_mask_area_filtered.tif",
        keep_area_mask.astype(np.uint8),
    )

    tiff.imwrite(
        out_dir / f"{mosaic_name}_mask_lifetime_classes.tif",
        final_lifetime_class_mask.astype(np.uint16),
    )

    tiff.imwrite(
        out_dir / f"{mosaic_name}_cell_mask_final.tif",
        final_cell_mask.astype(np.uint8),
    )

    if SAVE_DEBUG_CLASS_MASKS:
        tiff.imwrite(
            out_dir / f"{mosaic_name}_melanin_mask.tif",
            final_melanin_mask.astype(np.uint8),
        )

        tiff.imwrite(
            out_dir / f"{mosaic_name}_elastin_mask.tif",
            final_elastin_mask.astype(np.uint8),
        )

        tiff.imwrite(
            out_dir / f"{mosaic_name}_cell_low_confidence_mask.tif",
            final_low_confidence_cell_mask.astype(np.uint8),
        )

    tile_out_dir = out_dir / "tiles"
    tile_out_dir.mkdir(parents=True, exist_ok=True)

    cell_tiles = split_mosaic_to_tiles_snake(final_cell_mask, nrows, ncols)
    class_tiles = split_mosaic_to_tiles_snake(
        final_lifetime_class_mask,
        nrows,
        ncols,
    )

    for mask_path, cell_tile, class_tile in zip(mask_paths, cell_tiles, class_tiles):
        tile_number = extract_tile_number(mask_path.name)
        tile_label = f"{tile_number:05d}" if tile_number is not None else mask_path.stem

        tiff.imwrite(
            tile_out_dir / f"Im_{tile_label}_cell_mask_final.tif",
            cell_tile.astype(np.uint8),
        )

        tiff.imwrite(
            tile_out_dir / f"Im_{tile_label}_lifetime_classes.tif",
            class_tile.astype(np.uint16),
        )

    gmm_plot = out_dir / f"{mosaic_name}_phasor_gmm.png"

    plot_gmm_phasor(
        df=df,
        cluster_summary=cluster_summary,
        out_path=gmm_plot,
        title=(
            f"{visit_name} | {mosaic_name} | ROI mean phasor GMM\n"
            f"components={n_components}: {', '.join(expected_classes)} | "
            f"area ≥ {MIN_AREA_PX}px | cell probability ≥ {CELL_PROB_THRESHOLD}"
        ),
    )

    n_final_cells = int((df["final_class"] == "cell").sum())
    n_low_confidence = int((df["final_class"] == "cell_low_confidence").sum())

    print(f"[OK] ROIs after area filter: {len(df)}")
    print(f"     GMM components: {n_components} -> {expected_classes}")
    print(f"     Final cell ROIs: {n_final_cells}")
    print(f"     Low-confidence cell ROIs excluded: {n_low_confidence}")
    print(f"     ROI CSV: {roi_csv}")
    print(f"     Elastin CM: {elastin_csv}")
    print(f"     Final cell mask: {out_dir / f'{mosaic_name}_cell_mask_final.tif'}")
    print(f"     GMM plot: {gmm_plot}")


# =========================
# MAIN
# =========================


def main():
    if not PATIENT_DIR.exists():
        raise FileNotFoundError(f"No existe PATIENT_DIR:\n{PATIENT_DIR}")

    visit_dirs = sorted(
        [path for path in PATIENT_DIR.glob("visit*") if path.is_dir()],
        key=natural_key,
    )

    if len(visit_dirs) == 0:
        print("[WARN] No se encontraron carpetas visit*.")
        return

    print("\nGMM configuration by visit:")
    for visit_name, n_components in VISIT_GMM_COMPONENTS.items():
        print(
            f"  {visit_name}: {n_components} components -> "
            f"{get_classes_for_n_components(n_components)}"
        )
    print(
        f"  default: {DEFAULT_N_COMPONENTS} components -> "
        f"{get_classes_for_n_components(DEFAULT_N_COMPONENTS)}"
    )

    for visit_dir in visit_dirs:
        mosaic_dirs = sorted(
            [path for path in visit_dir.glob("Mosaic*") if path.is_dir()],
            key=natural_key,
        )

        for mosaic_dir in mosaic_dirs:
            try:
                process_mosaic(mosaic_dir)

            except Exception as exc:
                print(f"[ERROR] {mosaic_dir}")
                print(f"        {type(exc).__name__}: {exc}")

    print("\nListo.")


if __name__ == "__main__":
    main()
