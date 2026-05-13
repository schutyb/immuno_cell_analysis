#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate final reconstructed mosaic-level cells masks against manual tile masks.

Ground truth:
    <mosaic_dir>/random_forest/mask/immune_cells_mask_Im_00001.tiff

Prediction:
    <PATIENT_DIR>/analysis/global_gmm_green_corrected_from_area_masks_relabel/
        visitXX/MosaicXX/cells_mask.tif

The prediction is a full reconstructed mosaic mask.
For each manual tile, this script crops the corresponding tile region from
the reconstructed cells_mask.tif and compares tile vs tile.

Assumes 4x4 mosaic with snake scan reconstruction.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tiff
from PIL import Image
from skimage.measure import label
from skimage.transform import resize


# =========================
# CONFIG
# =========================

PATIENT_DIR = Path(
    "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449"
).expanduser()

MANUAL_MASK_SUBDIR = Path("random_forest/mask")

FINAL_MOSAIC_MASK_ROOT = (
    PATIENT_DIR / "analysis" / "global_gmm_green_corrected_from_area_masks_relabel"
)

FINAL_CELLS_MASK_NAME = "cells_mask.tif"

OUTPUT_DIR = (
    PATIENT_DIR
    / "segmentation_evaluation"
    / "global_gmm_green_corrected_cells_from_mosaic"
)

MASK_EXTENSIONS = [".png", ".tif", ".tiff"]

METHOD_NAME = "Global corrected green phasor GMM cells mask"

OBJECT_IOU_THRESHOLD = 0.10
FIG_DPI = 600

MOSAIC_ROWS = 4
MOSAIC_COLS = 4
SNAKE_PATTERN = True

ALL_METRICS = [
    "dice",
    "iou",
    "precision",
    "object_precision",
    "fp_objects",
    "false_positive_percentage",
    "relative_cell_count_error",
]


# =========================
# STYLE
# =========================

def set_nature_style():
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# =========================
# HELPERS
# =========================

def natural_key(path):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", str(path.name))
    ]


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


def read_mask(path):
    if path.suffix.lower() in [".tif", ".tiff"]:
        mask = tiff.imread(path)
    else:
        mask = np.array(Image.open(path))

    if mask.ndim == 3:
        mask = mask[..., 0]

    return mask > 0


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


def collect_mask_files(folder):
    if not folder.exists():
        return []

    files = []
    for extension in MASK_EXTENSIONS:
        files.extend(folder.glob(f"*{extension}"))

    return sorted(files, key=natural_key)


def tile_number_to_row_col(tile_number):
    """
    Convert 1-based tile number to mosaic row/col.
    For 4x4 snake pattern:

        1  2  3  4
        8  7  6  5
        9 10 11 12
       16 15 14 13
    """
    index = tile_number - 1
    row = index // MOSAIC_COLS
    col_linear = index % MOSAIC_COLS

    if SNAKE_PATTERN and row % 2 == 1:
        col = MOSAIC_COLS - 1 - col_linear
    else:
        col = col_linear

    return row, col


def crop_tile_from_mosaic_mask(mosaic_mask, tile_number, target_tile_shape):
    mosaic_h, mosaic_w = mosaic_mask.shape

    tile_h = mosaic_h // MOSAIC_ROWS
    tile_w = mosaic_w // MOSAIC_COLS

    row, col = tile_number_to_row_col(tile_number)

    y0 = row * tile_h
    y1 = y0 + tile_h
    x0 = col * tile_w
    x1 = x0 + tile_w

    crop = mosaic_mask[y0:y1, x0:x1]

    crop = resize_mask_if_needed(crop, target_tile_shape)

    return crop


def find_final_cells_mask(visit_name, mosaic_name):
    path = FINAL_MOSAIC_MASK_ROOT / visit_name / mosaic_name / FINAL_CELLS_MASK_NAME

    if path.exists():
        return path

    return None


# =========================
# METRICS
# =========================

def pixel_level_metrics(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    true_positive = np.logical_and(gt, pred).sum()
    false_positive = np.logical_and(~gt, pred).sum()
    false_negative = np.logical_and(gt, ~pred).sum()

    dice_denominator = 2 * true_positive + false_positive + false_negative
    iou_denominator = true_positive + false_positive + false_negative
    precision_denominator = true_positive + false_positive

    dice = 2 * true_positive / dice_denominator if dice_denominator > 0 else np.nan
    iou = true_positive / iou_denominator if iou_denominator > 0 else np.nan
    precision = true_positive / precision_denominator if precision_denominator > 0 else np.nan

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
    }


def object_level_metrics(gt, pred, iou_threshold):
    gt_labels = label(gt, connectivity=2)
    pred_labels = label(pred, connectivity=2)

    gt_ids = np.unique(gt_labels)
    pred_ids = np.unique(pred_labels)

    gt_ids = gt_ids[gt_ids != 0]
    pred_ids = pred_ids[pred_ids != 0]

    n_gt_objects = len(gt_ids)
    n_pred_objects = len(pred_ids)

    matched_gt = set()
    matched_pred = set()

    for pred_id in pred_ids:
        pred_object = pred_labels == pred_id

        best_iou = 0.0
        best_gt_id = None

        overlapping_gt_ids = np.unique(gt_labels[pred_object])
        overlapping_gt_ids = overlapping_gt_ids[overlapping_gt_ids != 0]

        for gt_id in overlapping_gt_ids:
            gt_object = gt_labels == gt_id

            intersection = np.logical_and(pred_object, gt_object).sum()
            union = np.logical_or(pred_object, gt_object).sum()

            iou = intersection / union if union > 0 else 0.0

            if iou > best_iou:
                best_iou = float(iou)
                best_gt_id = int(gt_id)

        if best_iou >= iou_threshold and best_gt_id is not None:
            matched_pred.add(int(pred_id))
            matched_gt.add(best_gt_id)

    tp_objects = len(matched_pred)
    fp_objects = n_pred_objects - tp_objects
    fn_objects = n_gt_objects - len(matched_gt)

    object_precision = (
        tp_objects / (tp_objects + fp_objects)
        if (tp_objects + fp_objects) > 0
        else np.nan
    )

    false_positive_percentage = (
        100.0 * fp_objects / (tp_objects + fp_objects)
        if (tp_objects + fp_objects) > 0
        else np.nan
    )

    cell_count_error = abs(n_pred_objects - n_gt_objects)

    relative_cell_count_error = (
        cell_count_error / n_gt_objects if n_gt_objects > 0 else np.nan
    )

    return {
        "n_gt_objects": int(n_gt_objects),
        "n_pred_objects": int(n_pred_objects),
        "tp_objects": int(tp_objects),
        "fp_objects": int(fp_objects),
        "fn_objects": int(fn_objects),
        "object_precision": float(object_precision),
        "cell_count_error": int(cell_count_error),
        "relative_cell_count_error": float(relative_cell_count_error),
        "false_positive_percentage": float(false_positive_percentage),
    }


def evaluate_single_prediction(gt, pred, meta):
    pred = resize_mask_if_needed(pred, gt.shape)

    pixel_metrics = pixel_level_metrics(gt, pred)
    object_metrics = object_level_metrics(gt, pred, OBJECT_IOU_THRESHOLD)

    return {
        **meta,
        "method": METHOD_NAME,
        "object_iou_threshold": OBJECT_IOU_THRESHOLD,
        **pixel_metrics,
        **object_metrics,
    }


# =========================
# QC OVERLAYS
# =========================

def save_overlay(gt, pred, output_path):
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    rgb = np.zeros((*gt.shape, 3), dtype=np.uint8)

    # manual GT = green
    rgb[..., 1][gt] = 180

    # prediction = red
    rgb[..., 0][pred] = 220

    # overlap = yellow
    overlap = gt & pred
    rgb[..., 0][overlap] = 255
    rgb[..., 1][overlap] = 255

    Image.fromarray(rgb).save(output_path)


# =========================
# PLOTS
# =========================

def metric_title(metric):
    titles = {
        "precision": "Pixel-level precision",
        "object_precision": "Object-level precision",
        "fp_objects": "False-positive objects / tile",
        "false_positive_percentage": "False-positive objects (%)",
        "relative_cell_count_error": "Relative cell count error",
        "dice": "Dice coefficient",
        "iou": "IoU / Jaccard",
    }

    return titles.get(metric, metric.replace("_", " "))


def metric_ylabel(metric):
    labels = {
        "precision": "Score",
        "object_precision": "Score",
        "false_positive_percentage": "% of predicted objects",
        "fp_objects": "Objects / tile",
        "relative_cell_count_error": "Relative error",
        "dice": "Score",
        "iou": "Score",
    }

    return labels.get(metric, metric_title(metric))


def add_mean_median_lines(axis, values, metric):
    mean_value = np.nanmean(values)
    median_value = np.nanmedian(values)

    if metric == "false_positive_percentage":
        mean_label = f"Mean = {mean_value:.1f}%"
        median_label = f"Median = {median_value:.1f}%"
    else:
        mean_label = f"Mean = {mean_value:.2f}"
        median_label = f"Median = {median_value:.2f}"

    axis.hlines(mean_value, 0.82, 1.18, colors="black", linewidth=1.6)
    axis.hlines(median_value, 0.82, 1.18, colors="black", linewidth=1.4, linestyles="--")

    axis.plot([], [], color="black", linewidth=1.6, label=mean_label)
    axis.plot([], [], color="black", linewidth=1.4, linestyle="--", label=median_label)


def plot_single_violin(axis, values, metric, seed=0):
    rng = np.random.default_rng(seed)

    violin = axis.violinplot(
        values,
        positions=[1],
        widths=0.65,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body in violin["bodies"]:
        body.set_facecolor("#D9D9D9")
        body.set_edgecolor("#6E6E6E")
        body.set_alpha(0.75)
        body.set_linewidth(0.8)

    x_values = 1 + rng.normal(0, 0.035, size=len(values))

    axis.scatter(
        x_values,
        values,
        s=18,
        color="#1F77B4",
        alpha=0.75,
        edgecolors="white",
        linewidths=0.35,
        zorder=3,
    )

    add_mean_median_lines(axis, values, metric)

    axis.set_xticks([1])
    axis.set_xticklabels([METHOD_NAME], rotation=20, ha="right")
    axis.set_title(metric_title(metric), pad=8)
    axis.set_ylabel(metric_ylabel(metric))
    axis.grid(axis="y", alpha=0.18, linewidth=0.6)

    if metric in ["precision", "object_precision", "dice", "iou"]:
        axis.set_ylim(0, 1.05)

    if metric == "false_positive_percentage":
        axis.set_ylim(0, 100)


def save_violin_plot(df, metric, output_path, ylim=None, seed=0):
    values = df[metric].dropna().astype(float).values

    if len(values) == 0:
        return

    fig, axis = plt.subplots(figsize=(3.2, 4.2), dpi=FIG_DPI)

    plot_single_violin(axis, values, metric, seed=seed)

    if ylim is not None:
        axis.set_ylim(*ylim)

    axis.legend(frameon=False, fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_two_violin_panel(df, output_path):
    metrics = [
        "object_precision",
        "false_positive_percentage",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 4.2), dpi=FIG_DPI)

    for index, (axis, metric) in enumerate(zip(axes, metrics)):
        values = df[metric].dropna().astype(float).values
        plot_single_violin(axis, values, metric, seed=index)
        axis.legend(frameon=False, fontsize=8, loc="best")

    fig.suptitle("Segmentation performance", fontsize=12, y=1.02)

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_key_metrics_panel(df, output_path):
    metrics = [
        "precision",
        "object_precision",
        "false_positive_percentage",
        "relative_cell_count_error",
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(12.5, 4.0), dpi=FIG_DPI)

    for index, (axis, metric) in enumerate(zip(axes, metrics)):
        values = df[metric].dropna().astype(float).values
        plot_single_violin(axis, values, metric, seed=index)

        if index == 0:
            axis.legend(frameon=False, fontsize=8, loc="best")
        else:
            legend = axis.get_legend()
            if legend is not None:
                legend.remove()

    fig.suptitle(
        "Final cells mask performance against manual tile annotations",
        fontsize=12,
        y=1.05,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_performance_cards(df, output_path):
    metrics = [
        "precision",
        "object_precision",
        "false_positive_percentage",
        "relative_cell_count_error",
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(12.5, 3.2), dpi=FIG_DPI)

    for axis, metric in zip(axes, metrics):
        values = df[metric].dropna().astype(float).values

        mean_value = np.nanmean(values)
        std_value = np.nanstd(values, ddof=1)
        median_value = np.nanmedian(values)
        n_values = len(values)

        axis.axis("off")

        axis.text(0.5, 0.78, metric_title(metric), ha="center", va="center",
                  fontsize=9, fontweight="bold")

        value_text = f"{mean_value:.2f}"
        if metric == "false_positive_percentage":
            value_text += "%"

        axis.text(0.5, 0.50, value_text, ha="center", va="center",
                  fontsize=26, fontweight="bold")

        axis.text(0.5, 0.31, f"mean ± SD: {mean_value:.2f} ± {std_value:.2f}",
                  ha="center", va="center", fontsize=8)

        axis.text(0.5, 0.20, f"median: {median_value:.2f} | n={n_values}",
                  ha="center", va="center", fontsize=8)

        axis.add_patch(
            plt.Rectangle(
                (0.04, 0.08),
                0.92,
                0.82,
                fill=False,
                linewidth=0.8,
                alpha=0.65,
                transform=axis.transAxes,
            )
        )

    fig.suptitle(f"Final cells segmentation performance: {METHOD_NAME}", fontsize=12)

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# =========================
# README
# =========================

def save_readme(summary_df, output_path):
    lines = []

    lines.append("README - Final cells mask evaluation from reconstructed mosaics\n")
    lines.append("=" * 65 + "\n\n")

    lines.append("Method evaluated\n")
    lines.append("----------------\n")
    lines.append(
        f"{METHOD_NAME}: global ROI-level GMM classification in corrected green "
        "phasor space. The evaluated prediction is the final reconstructed "
        "mosaic-level cells_mask.tif cropped back to manually annotated tiles.\n\n"
    )

    lines.append("Ground truth\n")
    lines.append("------------\n")
    lines.append(
        "Manual binary immune-cell masks stored per tile in random_forest/mask.\n\n"
    )

    lines.append("Prediction\n")
    lines.append("----------\n")
    lines.append(
        "Mosaic-level cells_mask.tif from the global corrected phasor GMM output. "
        "For each manually annotated tile, the corresponding tile crop was "
        "extracted from the reconstructed mosaic mask using the 4x4 snake pattern.\n\n"
    )

    lines.append("Object matching\n")
    lines.append("---------------\n")
    lines.append(
        f"A predicted object is considered a true-positive object when its best "
        f"IoU overlap with a manual object is >= {OBJECT_IOU_THRESHOLD}.\n\n"
    )

    lines.append("Summary values\n")
    lines.append("--------------\n")

    for _, row in summary_df.iterrows():
        lines.append(
            f"{row['metric_label']}: "
            f"mean={row['mean']:.4f}, "
            f"std={row['std']:.4f}, "
            f"median={row['median']:.4f}, "
            f"n={int(row['count'])}\n"
        )

    output_path.write_text("".join(lines))


# =========================
# MAIN
# =========================

def main():
    set_nature_style()

    if not PATIENT_DIR.exists():
        raise FileNotFoundError(f"No existe PATIENT_DIR:\n{PATIENT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    overlay_dir = OUTPUT_DIR / "overlays_manual_vs_cells_mask"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    visit_dirs = sorted(
        [path for path in PATIENT_DIR.glob("visit*") if path.is_dir()],
        key=natural_key,
    )

    if len(visit_dirs) == 0:
        print("[WARN] No se encontraron carpetas visit*.")
        return

    for visit_dir in visit_dirs:
        visit_name = visit_dir.name

        mosaic_dirs = sorted(
            [path for path in visit_dir.glob("Mosaic*") if path.is_dir()],
            key=natural_key,
        )

        for mosaic_dir in mosaic_dirs:
            mosaic_name = mosaic_dir.name

            manual_dir = mosaic_dir / MANUAL_MASK_SUBDIR
            manual_files = collect_mask_files(manual_dir)

            if len(manual_files) == 0:
                continue

            final_cells_mask_path = find_final_cells_mask(visit_name, mosaic_name)

            print(f"\n[PROCESS] {visit_name} | {mosaic_name}")
            print(f"Manual masks: {len(manual_files)}")

            if final_cells_mask_path is None:
                print("[WARN] Missing reconstructed cells_mask.tif")
                continue

            print(f"Final mosaic mask: {final_cells_mask_path}")

            final_mosaic_mask = read_mask(final_cells_mask_path)

            for manual_file in manual_files:
                tile_number = extract_tile_number(manual_file.name)

                if tile_number is None:
                    print(f"[WARN] Could not extract tile number: {manual_file}")
                    continue

                gt = read_mask(manual_file)

                pred_crop = crop_tile_from_mosaic_mask(
                    final_mosaic_mask,
                    tile_number,
                    gt.shape,
                )

                meta = {
                    "visit": visit_name,
                    "mosaic": mosaic_name,
                    "tile": tile_number,
                    "manual_mask": str(manual_file),
                    "final_mosaic_mask": str(final_cells_mask_path),
                }

                row = evaluate_single_prediction(gt, pred_crop, meta)
                rows.append(row)

                overlay_name = (
                    f"{visit_name}_{mosaic_name}_tile{tile_number:02d}"
                    "_manual_green_pred_red_overlap_yellow.png"
                )

                save_overlay(
                    gt,
                    pred_crop,
                    overlay_dir / overlay_name,
                )

    if len(rows) == 0:
        print("\nNo evaluation rows generated.")
        return

    df = pd.DataFrame(rows)

    keep_cols = [
        "visit",
        "mosaic",
        "tile",
        "method",
        "object_iou_threshold",
        "manual_mask",
        "final_mosaic_mask",
        "dice",
        "iou",
        "precision",
        "object_precision",
        "fp_objects",
        "false_positive_percentage",
        "relative_cell_count_error",
        "n_gt_objects",
        "n_pred_objects",
        "tp_objects",
        "fn_objects",
        "cell_count_error",
    ]

    keep_cols = [column for column in keep_cols if column in df.columns]
    df = df[keep_cols]

    summary = (
        df[ALL_METRICS]
        .agg(["mean", "std", "median", "count"])
        .T.reset_index()
        .rename(columns={"index": "metric"})
    )

    summary["metric_label"] = summary["metric"].map(metric_title)

    tile_csv = OUTPUT_DIR / "segmentation_key_metrics_by_tile.csv"
    summary_csv = OUTPUT_DIR / "segmentation_metrics_summary.csv"

    df.to_csv(tile_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    save_key_metrics_panel(
        df,
        OUTPUT_DIR / "final_cells_key_metrics_panel.png",
    )

    save_two_violin_panel(
        df,
        OUTPUT_DIR / "final_cells_object_precision_and_fp_percentage.png",
    )

    save_performance_cards(
        df,
        OUTPUT_DIR / "final_cells_performance_cards.png",
    )

    save_violin_plot(
        df,
        "object_precision",
        OUTPUT_DIR / "violin_object_precision.png",
        ylim=(0, 1.05),
    )

    save_violin_plot(
        df,
        "false_positive_percentage",
        OUTPUT_DIR / "violin_false_positive_percentage.png",
        ylim=(0, 100),
    )

    save_readme(
        summary,
        OUTPUT_DIR / "README_segmentation_evaluation.txt",
    )

    print("\n[DONE]")
    print(f"Tile metrics:\n{tile_csv}")
    print(f"Summary:\n{summary_csv}")
    print(f"Overlays:\n{overlay_dir}")
    print(f"Plots and README saved in:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()