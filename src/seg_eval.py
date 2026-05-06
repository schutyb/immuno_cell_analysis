#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate final U-Net + FLIM segmentation against manual cell masks.

Final method:
    U-Net + FLIM = U-Net candidate segmentation + area filtering
    + ROI-level FLIM phasor/lifetime filtering.

Main metrics:
    - Pixel-level precision
    - Object-level precision
    - False-positive objects (% of predicted objects)
    - Relative cell count error

Secondary metrics:
    - Dice
    - IoU
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import tifffile as tiff
from PIL import Image

import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.transform import resize


# =========================
# CONFIG
# =========================

PATIENT_DIR = Path(
    "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449"
).expanduser()

MANUAL_MASK_SUBDIR = Path("random_forest/mask")
FINAL_MASK_SUBDIR = Path("segmentation_area_phasor/tiles")

OUTPUT_DIR = PATIENT_DIR / "segmentation_evaluation" / "unet_flim_final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MASK_EXTENSIONS = [".png", ".tif", ".tiff"]

METHOD_NAME = "U-Net + FLIM"

OBJECT_IOU_THRESHOLD = 0.10

FIG_DPI = 600

PRIMARY_METRICS = [
    "precision",
    "object_precision",
    "false_positive_percentage",
    "relative_cell_count_error",
]

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
# PLOT STYLE
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
        int(t) if t.isdigit() else t.lower()
        for t in re.split(r"(\d+)", str(path.name))
    ]


def extract_tile_number(name):
    patterns = [
        r"Im_(\d+)",
        r"_t(\d+)",
        r"tile[_-]?(\d+)",
    ]

    for pat in patterns:
        m = re.search(pat, name, re.IGNORECASE)
        if m:
            return int(m.group(1))

    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else None


def read_mask(path):
    if path.suffix.lower() in [".tif", ".tiff"]:
        arr = tiff.imread(path)
    else:
        arr = np.array(Image.open(path))

    if arr.ndim == 3:
        arr = arr[..., 0]

    return arr > 0


def resize_mask_if_needed(mask, target_shape):
    if mask.shape == target_shape:
        return mask

    mask_rs = resize(
        mask.astype(float),
        target_shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )

    return mask_rs > 0.5


def collect_mask_files(folder):
    if not folder.exists():
        return []

    files = []
    for ext in MASK_EXTENSIONS:
        files.extend(folder.glob(f"*{ext}"))

    return sorted(files, key=natural_key)


def match_by_tile(tile_number, files):
    for f in files:
        if extract_tile_number(f.name) == tile_number:
            return f
    return None


# =========================
# METRICS
# =========================

def pixel_level_metrics(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    tp = np.logical_and(gt, pred).sum()
    fp = np.logical_and(~gt, pred).sum()
    fn = np.logical_and(gt, ~pred).sum()

    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else np.nan
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
    }


def object_level_metrics(gt, pred, iou_threshold):
    gt_lab = label(gt, connectivity=2)
    pred_lab = label(pred, connectivity=2)

    gt_ids = np.unique(gt_lab)
    pred_ids = np.unique(pred_lab)

    gt_ids = gt_ids[gt_ids != 0]
    pred_ids = pred_ids[pred_ids != 0]

    n_gt = len(gt_ids)
    n_pred = len(pred_ids)

    matched_gt = set()
    matched_pred = set()

    for pred_id in pred_ids:
        pred_obj = pred_lab == pred_id

        best_iou = 0.0
        best_gt_id = None

        overlapping_gt_ids = np.unique(gt_lab[pred_obj])
        overlapping_gt_ids = overlapping_gt_ids[overlapping_gt_ids != 0]

        for gt_id in overlapping_gt_ids:
            gt_obj = gt_lab == gt_id

            intersection = np.logical_and(pred_obj, gt_obj).sum()
            union = np.logical_or(pred_obj, gt_obj).sum()

            iou = intersection / union if union > 0 else 0.0

            if iou > best_iou:
                best_iou = float(iou)
                best_gt_id = int(gt_id)

        if best_iou >= iou_threshold:
            matched_pred.add(int(pred_id))
            matched_gt.add(best_gt_id)

    tp_objects = len(matched_pred)
    fp_objects = n_pred - tp_objects
    fn_objects = n_gt - len(matched_gt)

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

    cell_count_error = abs(n_pred - n_gt)
    relative_cell_count_error = cell_count_error / n_gt if n_gt > 0 else np.nan

    return {
        "n_gt_objects": int(n_gt),
        "n_pred_objects": int(n_pred),
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

    pix = pixel_level_metrics(gt, pred)
    obj = object_level_metrics(gt, pred, OBJECT_IOU_THRESHOLD)

    return {
        **meta,
        "method": METHOD_NAME,
        "object_iou_threshold": OBJECT_IOU_THRESHOLD,
        **pix,
        **obj,
    }


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


def add_mean_median_lines(ax, values):
    mean = np.nanmean(values)
    median = np.nanmedian(values)

    ax.hlines(
        mean,
        0.82,
        1.18,
        colors="black",
        linewidth=1.6,
        linestyles="-",
        zorder=4,
        label="Mean",
    )

    ax.hlines(
        median,
        0.82,
        1.18,
        colors="black",
        linewidth=1.4,
        linestyles="--",
        zorder=4,
        label="Median",
    )


def plot_single_violin(ax, values, metric, seed=0):
    rng = np.random.default_rng(seed)

    violin = ax.violinplot(
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

    x = 1 + rng.normal(0, 0.035, size=len(values))

    ax.scatter(
        x,
        values,
        s=18,
        color="#1F77B4",
        alpha=0.75,
        edgecolors="white",
        linewidths=0.35,
        zorder=3,
    )

    add_mean_median_lines(ax, values)

    ax.set_xticks([1])
    ax.set_xticklabels([METHOD_NAME])
    ax.set_title(metric_title(metric), pad=8)
    ax.set_ylabel(metric_ylabel(metric))
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)

    if metric in ["precision", "object_precision"]:
        ax.set_ylim(0, 1.05)

    if metric == "false_positive_percentage":
        ax.set_ylim(0, 100)


def save_violin_plot(df, metric, out_path, ylim=None, seed=0):
    values = df[metric].dropna().astype(float).values

    if len(values) == 0:
        return

    fig, ax = plt.subplots(figsize=(3.2, 4.2), dpi=FIG_DPI)

    plot_single_violin(ax, values, metric, seed=seed)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.legend(
        frameon=False,
        fontsize=8,
        loc="best",
    )

    fig.tight_layout()

    fig.savefig(
        out_path,
        dpi=FIG_DPI,
        bbox_inches="tight",
    )

    fig.savefig(
        out_path.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    plt.close(fig)


def save_two_violin_panel(df, out_path):
    metrics = [
        "object_precision",
        "false_positive_percentage",
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6.4, 4.2),
        dpi=FIG_DPI,
    )

    for i, (ax, metric) in enumerate(zip(axes, metrics)):
        values = df[metric].dropna().astype(float).values
        plot_single_violin(ax, values, metric, seed=i)

        ax.legend(
            frameon=False,
            fontsize=8,
            loc="best",
        )

    fig.suptitle(
        "Segmentation performance",
        fontsize=12,
        y=1.02,
    )

    fig.tight_layout()

    fig.savefig(
        out_path,
        dpi=FIG_DPI,
        bbox_inches="tight",
    )

    fig.savefig(
        out_path.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    plt.close(fig)


def save_key_metrics_panel(df, out_path):
    metrics = [
        "precision",
        "object_precision",
        "false_positive_percentage",
        "relative_cell_count_error",
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(12.5, 4.0), dpi=FIG_DPI)

    for i, (ax, metric) in enumerate(zip(axes, metrics)):
        values = df[metric].dropna().astype(float).values
        plot_single_violin(ax, values, metric, seed=i)

    axes[0].legend(
        frameon=False,
        fontsize=8,
        loc="best",
    )

    fig.suptitle(
        "Final segmentation performance against manual annotations",
        fontsize=12,
        y=1.05,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_performance_cards(df, out_path):
    metrics = [
        "precision",
        "object_precision",
        "false_positive_percentage",
        "relative_cell_count_error",
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(12.5, 3.2), dpi=FIG_DPI)

    for ax, metric in zip(axes, metrics):
        values = df[metric].dropna().astype(float).values
        mean = np.nanmean(values)
        std = np.nanstd(values, ddof=1)
        median = np.nanmedian(values)
        n = len(values)

        ax.axis("off")

        ax.text(
            0.5,
            0.78,
            metric_title(metric),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

        value_text = f"{mean:.2f}"
        if metric == "false_positive_percentage":
            value_text += "%"

        ax.text(
            0.5,
            0.50,
            value_text,
            ha="center",
            va="center",
            fontsize=26,
            fontweight="bold",
        )

        ax.text(
            0.5,
            0.31,
            f"mean ± SD: {mean:.2f} ± {std:.2f}",
            ha="center",
            va="center",
            fontsize=8,
        )

        ax.text(
            0.5,
            0.20,
            f"median: {median:.2f} | n={n}",
            ha="center",
            va="center",
            fontsize=8,
        )

        ax.add_patch(
            plt.Rectangle(
                (0.04, 0.08),
                0.92,
                0.82,
                fill=False,
                linewidth=0.8,
                alpha=0.65,
                transform=ax.transAxes,
            )
        )

    fig.suptitle("Final segmentation performance: U-Net + FLIM", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# =========================
# README
# =========================

def save_readme(summary_df, out_path):
    lines = []

    lines.append("README - Final segmentation evaluation\n")
    lines.append("=" * 45 + "\n\n")

    lines.append("Method evaluated\n")
    lines.append("----------------\n")
    lines.append(
        "U-Net + FLIM: candidate cell segmentation using U-Net, followed by "
        "area filtering and ROI-level FLIM phasor/lifetime filtering.\n\n"
    )

    lines.append("Ground truth\n")
    lines.append("------------\n")
    lines.append(
        "Manual binary masks containing only immune cells were used as ground truth.\n\n"
    )

    lines.append("Object matching\n")
    lines.append("---------------\n")
    lines.append(
        f"A predicted object is considered a true-positive object when its best "
        f"IoU overlap with a manual object is >= {OBJECT_IOU_THRESHOLD}.\n\n"
    )

    lines.append("Main metrics\n")
    lines.append("------------\n")
    lines.append(
        "Pixel-level precision:\n"
        "    Fraction of predicted cell pixels that overlap with manual cell pixels.\n"
        "    Formula: TP_pixels / (TP_pixels + FP_pixels).\n\n"
    )
    lines.append(
        "Object-level precision:\n"
        "    Fraction of predicted connected components that match true manually "
        "annotated cells.\n"
        "    Formula: TP_objects / (TP_objects + FP_objects).\n\n"
    )
    lines.append(
        "False-positive objects (%):\n"
        "    Percentage of predicted objects that were false positives.\n"
        "    Formula: 100 * FP_objects / (TP_objects + FP_objects).\n"
        "    This value is bounded between 0 and 100%.\n\n"
    )
    lines.append(
        "Relative cell count error:\n"
        "    Absolute difference between predicted and manual cell counts, normalized "
        "by the manual count.\n"
        "    Formula: |N_pred - N_GT| / N_GT.\n\n"
    )
    lines.append(
        "Dice and IoU:\n"
        "    Secondary pixel-overlap metrics included for reference if requested.\n\n"
    )

    lines.append("Files generated\n")
    lines.append("---------------\n")
    lines.append("unet_flim_metrics_by_tile.csv\n")
    lines.append("unet_flim_metrics_summary.csv\n")
    lines.append("unet_flim_key_metrics_panel.png / .pdf\n")
    lines.append("unet_flim_object_precision_and_fp_percentage.png / .pdf\n")
    lines.append("unet_flim_performance_cards.png / .pdf\n")
    lines.append("violin_object_precision.png / .pdf\n")
    lines.append("violin_false_positive_percentage.png / .pdf\n\n")

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

    out_path.write_text("".join(lines))


# =========================
# MAIN
# =========================

def main():
    set_nature_style()

    rows = []

    visit_dirs = sorted(
        [p for p in PATIENT_DIR.glob("visit*") if p.is_dir()],
        key=natural_key,
    )

    for visit_dir in visit_dirs:
        visit_name = visit_dir.name

        mosaic_dirs = sorted(
            [p for p in visit_dir.glob("Mosaic*") if p.is_dir()],
            key=natural_key,
        )

        for mosaic_dir in mosaic_dirs:
            mosaic_name = mosaic_dir.name

            manual_dir = mosaic_dir / MANUAL_MASK_SUBDIR
            final_dir = mosaic_dir / FINAL_MASK_SUBDIR

            manual_files = collect_mask_files(manual_dir)

            if len(manual_files) == 0:
                continue

            final_files = collect_mask_files(final_dir)

            print(f"\n[PROCESS] {visit_name} | {mosaic_name}")
            print(f"Manual masks: {len(manual_files)}")

            for manual_file in manual_files:
                tile_number = extract_tile_number(manual_file.name)

                if tile_number is None:
                    print(f"[WARNING] Could not extract tile number: {manual_file}")
                    continue

                final_file = match_by_tile(tile_number, final_files)

                if final_file is None:
                    print(f"[WARNING] Missing final mask for tile {tile_number}")
                    continue

                gt = read_mask(manual_file)
                pred = read_mask(final_file)

                meta = {
                    "visit": visit_name,
                    "mosaic": mosaic_name,
                    "tile": tile_number,
                    "manual_mask": str(manual_file),
                    "final_mask": str(final_file),
                }

                row = evaluate_single_prediction(gt, pred, meta)
                rows.append(row)

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
        "final_mask",
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
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    summary = (
        df[ALL_METRICS]
        .agg(["mean", "std", "median", "count"])
        .T
        .reset_index()
        .rename(columns={"index": "metric"})
    )

    summary["metric_label"] = summary["metric"].map(metric_title)

    tile_csv = OUTPUT_DIR / "unet_flim_metrics_by_tile.csv"
    summary_csv = OUTPUT_DIR / "unet_flim_metrics_summary.csv"

    df.to_csv(tile_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    save_key_metrics_panel(
        df,
        OUTPUT_DIR / "unet_flim_key_metrics_panel.png",
    )

    save_two_violin_panel(
        df,
        OUTPUT_DIR / "unet_flim_object_precision_and_fp_percentage.png",
    )

    save_performance_cards(
        df,
        OUTPUT_DIR / "unet_flim_performance_cards.png",
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
    print(f"Plots and README saved in:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()