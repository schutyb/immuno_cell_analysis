#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarize final segmentation performance for the selected pipeline.

This script reads tile-level segmentation metrics and extracts only the final
method used for downstream immuno-cell analysis.

The goal is not to compare multiple segmentation algorithms, but to report the
performance of the final selected pipeline against manual expert masks.

Expected input:
    segmentation_key_metrics_by_tile.csv

Expected columns:
    method
    precision
    object_precision
    fp_objects
    relative_cell_count_error

Optional columns:
    dice
    iou

Outputs:
    - tile-level CSV for the final method only
    - summary CSV with mean, std, median, and count
    - boxplot panel of key metrics
    - mean ± SD barplot
    - text explanation for manuscript/report writing

How to use:
    1. Edit EVAL_DIR.
    2. Edit FINAL_METHOD_NAME_IN_CSV if the method name changes.
    3. Run:

        python -m src.segmentation.summarize_final_segmentation_evaluation

    or directly:

        python src/segmentation/summarize_final_segmentation_evaluation.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================

EVAL_DIR = Path(
    "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449/"
    "segmentation_evaluation"
).expanduser()

TILE_CSV = EVAL_DIR / "segmentation_key_metrics_by_tile.csv"

OUTPUT_DIR = EVAL_DIR / "final_unet_flim_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_DPI = 600

FINAL_METHOD_NAME_IN_CSV = "PeterChang_area_FLIM"
FINAL_METHOD_LABEL = "U-Net + FLIM"

KEY_METRICS = [
    "precision",
    "object_precision",
    "fp_objects",
    "relative_cell_count_error",
]

OPTIONAL_METRICS = [
    "dice",
    "iou",
]

METRIC_LABELS = {
    "precision": "Pixel-level precision",
    "object_precision": "Object-level precision",
    "fp_objects": "False-positive objects per tile",
    "relative_cell_count_error": "Relative cell count error",
    "dice": "Dice coefficient",
    "iou": "IoU / Jaccard",
}


# =========================
# HELPERS
# =========================


def metric_title(metric):
    return METRIC_LABELS.get(metric, metric.replace("_", " "))


def load_final_method_only():
    if not TILE_CSV.exists():
        raise FileNotFoundError(f"Could not find CSV:\n{TILE_CSV}")

    df = pd.read_csv(TILE_CSV)

    if "method" not in df.columns:
        raise ValueError("CSV does not contain 'method' column.")

    df = df[df["method"] == FINAL_METHOD_NAME_IN_CSV].copy()

    if len(df) == 0:
        raise ValueError(
            f"Could not find rows for method='{FINAL_METHOD_NAME_IN_CSV}' "
            f"in:\n{TILE_CSV}"
        )

    df["method_label"] = FINAL_METHOD_LABEL

    return df


def build_summary(df):
    metrics = [m for m in KEY_METRICS + OPTIONAL_METRICS if m in df.columns]

    summary = df[metrics].agg(["mean", "std", "median", "count"]).T
    summary = summary.reset_index().rename(columns={"index": "metric"})

    summary["metric_label"] = summary["metric"].map(metric_title)
    summary["method"] = FINAL_METHOD_LABEL

    summary = summary[
        [
            "method",
            "metric",
            "metric_label",
            "mean",
            "std",
            "median",
            "count",
        ]
    ]

    return summary


def save_boxplot_panel(df, out_path):
    metrics = [m for m in KEY_METRICS if m in df.columns]

    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(4.3 * len(metrics), 5.0),
    )

    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        values = df[metric].dropna().values

        ax.boxplot(
            [values],
            labels=[FINAL_METHOD_LABEL],
            showmeans=True,
            meanprops={
                "marker": "o",
                "markerfacecolor": "black",
                "markeredgecolor": "black",
                "markersize": 5,
            },
        )

        ax.set_title(metric_title(metric), fontsize=10)
        ax.grid(axis="y", alpha=0.25)

        if metric in ["precision", "object_precision"]:
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Score")
        elif metric == "relative_cell_count_error":
            ax.set_ylabel("Relative error")
        elif metric == "fp_objects":
            ax.set_ylabel("Objects / tile")
        else:
            ax.set_ylabel(metric_title(metric))

    fig.suptitle(
        "Final segmentation performance: U-Net + FLIM",
        fontsize=14,
        y=1.02,
    )

    fig.tight_layout()
    fig.savefig(
        out_path,
        dpi=FIG_DPI,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)


def save_summary_barplot(df, out_path):
    """
    Small compact figure showing mean ± SD for the 4 metrics.
    """
    metrics = [m for m in KEY_METRICS if m in df.columns]

    means = [df[m].mean() for m in metrics]
    stds = [df[m].std() for m in metrics]
    labels = [metric_title(m) for m in metrics]

    plt.figure(figsize=(9, 5))

    x = np.arange(len(metrics))
    plt.bar(x, means, yerr=stds, capsize=5, alpha=0.85)

    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Metric value")
    plt.title("U-Net + FLIM segmentation performance | mean ± SD")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


def save_explanation_txt(df, summary_df, out_path):
    lines = []

    lines.append("Final segmentation evaluation: U-Net + FLIM\n")
    lines.append("=" * 50 + "\n\n")

    lines.append(
        "This evaluation validates the final segmentation pipeline used for "
        "downstream immune-cell analysis. The goal of this paper is not to compare "
        "segmentation algorithms, but to confirm that the final U-Net + FLIM "
        "pipeline provides biologically meaningful immune-cell masks when compared "
        "against manual expert annotations.\n\n"
    )

    lines.append(
        "Final method evaluated:\n"
        "- U-Net + FLIM: U-Net candidate segmentation followed by area filtering "
        "and ROI-level FLIM phasor/lifetime filtering.\n\n"
    )

    explanations = {
        "precision": (
            "Pixel-level precision measures the fraction of all pixels predicted "
            "as cell that are also labeled as cell in the manual expert mask. "
            "It is calculated as TP / (TP + FP). Higher values indicate that fewer "
            "non-cell pixels are included in the final segmentation."
        ),
        "object_precision": (
            "Object-level precision measures the fraction of detected connected "
            "components that correspond to true manually annotated cells. A predicted "
            "object is considered correct when it sufficiently overlaps a manual "
            "cell object. This metric is central for this study because the biological "
            "goal is to identify true immune-cell candidates rather than perfectly "
            "reconstruct cell boundaries."
        ),
        "fp_objects": (
            "False-positive objects per tile measures the number of predicted cell "
            "objects that do not match any manual cell annotation. Lower values "
            "indicate fewer non-cellular structures, such as elastin, melanin, or "
            "noise, being incorrectly retained as cells."
        ),
        "relative_cell_count_error": (
            "Relative cell count error measures the absolute difference between "
            "the predicted and manual cell counts, normalized by the manual count: "
            "|N_pred - N_GT| / N_GT. Lower values indicate more accurate cell "
            "quantification, which is essential for downstream cell-density and "
            "longitudinal immune-response analyses."
        ),
        "dice": (
            "Dice coefficient is a secondary pixel-overlap metric. It reflects "
            "boundary-level agreement between predicted and manual masks. In this "
            "project it is not the main endpoint because the primary goal is "
            "accurate immune-cell detection and quantification rather than perfect "
            "boundary reconstruction."
        ),
        "iou": (
            "IoU/Jaccard is another secondary pixel-overlap metric. Like Dice, it "
            "is useful for reference but less central here than precision, false "
            "positives, and cell-count error."
        ),
    }

    for metric in KEY_METRICS + OPTIONAL_METRICS:
        if metric not in summary_df["metric"].values:
            continue

        row = summary_df[summary_df["metric"] == metric].iloc[0]

        lines.append(f"{metric_title(metric)}\n")
        lines.append("-" * len(metric_title(metric)) + "\n")
        lines.append(explanations[metric] + "\n\n")
        lines.append(
            f"Result for U-Net + FLIM: "
            f"mean = {row['mean']:.4f}, "
            f"std = {row['std']:.4f}, "
            f"median = {row['median']:.4f}, "
            f"n = {int(row['count'])} tiles.\n\n"
        )

    lines.append("Suggested interpretation\n")
    lines.append("-" * 24 + "\n")
    lines.append(
        "These metrics should be presented as validation of the final segmentation "
        "pipeline. Dice and IoU may be reported as supplementary or secondary "
        "overlap metrics if requested, but the main paper should emphasize "
        "precision, object-level precision, false-positive objects per tile, and "
        "relative cell count error because these directly support the downstream "
        "biological objective: reliable immune-cell detection and quantification.\n"
    )

    out_path.write_text("".join(lines))


# =========================
# MAIN
# =========================


def main():
    df = load_final_method_only()

    print(f"Loaded: {TILE_CSV}")
    print(f"Rows for {FINAL_METHOD_LABEL}: {len(df)}")
    print(f"Output dir: {OUTPUT_DIR}")

    keep_cols = [
        c
        for c in [
            "visit",
            "mosaic",
            "tile",
            "method",
            "method_label",
            *KEY_METRICS,
            *OPTIONAL_METRICS,
            "object_iou_threshold",
            "n_gt_objects",
            "n_pred_objects",
            "tp_objects",
            "fp_objects",
            "fn_objects",
            "cell_count_error",
        ]
        if c in df.columns
    ]

    by_tile_csv = OUTPUT_DIR / "unet_flim_key_metrics_by_tile.csv"
    df[keep_cols].to_csv(by_tile_csv, index=False)

    summary_df = build_summary(df)
    summary_csv = OUTPUT_DIR / "unet_flim_key_metrics_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    fig_box = OUTPUT_DIR / "unet_flim_key_metrics_boxplots.png"
    save_boxplot_panel(df, fig_box)

    fig_bar = OUTPUT_DIR / "unet_flim_key_metrics_mean_sd.png"
    save_summary_barplot(df, fig_bar)

    txt_path = OUTPUT_DIR / "unet_flim_evaluation_explanation.txt"
    save_explanation_txt(df, summary_df, txt_path)

    print("\nDone.")
    print(f"Tile-level metrics:\n{by_tile_csv}")
    print(f"Summary:\n{summary_csv}")
    print(f"Boxplot figure:\n{fig_box}")
    print(f"Mean ± SD figure:\n{fig_bar}")
    print(f"Explanation:\n{txt_path}")


if __name__ == "__main__":
    main()
