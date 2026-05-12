#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze tile-level segmentation metrics as a function of imaging depth.

This script reads the final segmentation evaluation CSV and adds z-depth
information extracted from the mosaic name.

It analyzes:
    - object-level precision vs imaging depth
    - false-positive percentage vs imaging depth
    - manual cell counts vs imaging depth
    - predicted cell counts vs imaging depth

Expected input:
    <patient_dir>/segmentation_evaluation/final_area_phasor_gmm/
        segmentation_key_metrics_by_tile.csv

Outputs:
    <patient_dir>/segmentation_evaluation/depth_analysis/
        tile_metrics_with_depth.csv
        scatter_*_vs_depth.png / .pdf
        boxplot_*_by_depth_group.png / .pdf
        summary_panel_by_depth.png / .pdf
        depth_group_summary.csv

How to use:
    1. Run evaluate_final_masks_against_manual.py first.
    2. Edit PATIENT_DIR.
    3. Run:

        python -m src.segmentation.analyze_segmentation_metrics_by_depth

    or directly:

        python src/segmentation/analyze_segmentation_metrics_by_depth.py
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =========================================================
# CONFIG
# =========================================================

PATIENT_DIR = Path(
    "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449"
).expanduser()

CSV_PATH = (
    PATIENT_DIR
    / "segmentation_evaluation"
    / "final_area_phasor_gmm"
    / "segmentation_key_metrics_by_tile.csv"
)

OUTPUT_DIR = PATIENT_DIR / "segmentation_evaluation" / "depth_analysis"

FIG_DPI = 600

DEPTH_GROUP_BINS = {
    "superficial": (None, 110),
    "mid": (110, 135),
    "deep": (135, None),
}


# =========================================================
# STYLE
# =========================================================


def set_style():
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# =========================================================
# HELPERS
# =========================================================


def extract_z_depth(mosaic_name):
    """
    Extract z depth from names such as:
        Mosaic03_4x4_FOV600_z110_32Sp
    """
    match = re.search(r"_z(\d+)", str(mosaic_name))

    if match:
        return int(match.group(1))

    return np.nan


def assign_depth_group(z_depth):
    """
    Assign a z-depth group.

    Default groups:
        superficial: z <= 110
        mid:         110 < z <= 135
        deep:        z > 135
    """
    if not np.isfinite(z_depth):
        return "unknown"

    if z_depth <= 110:
        return "superficial"

    if z_depth <= 135:
        return "mid"

    return "deep"


def metric_title(metric):
    mapping = {
        "object_precision": "Object-level precision",
        "false_positive_percentage": "False-positive objects (%)",
        "n_gt_objects": "Manual cell count",
        "n_pred_objects": "Predicted cell count",
        "precision": "Pixel-level precision",
        "relative_cell_count_error": "Relative cell count error",
    }

    return mapping.get(metric, metric.replace("_", " "))


def validate_columns(df, required_columns):
    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise ValueError(
            "Missing required columns in input CSV:\n"
            + "\n".join(f"- {column}" for column in missing)
        )


# =========================================================
# PLOTS
# =========================================================


def save_scatter(df, metric, output_path):
    fig, axis = plt.subplots(figsize=(5.5, 4.5), dpi=FIG_DPI)

    visits = sorted(df["visit"].dropna().unique())

    for visit in visits:
        sub_df = df[df["visit"] == visit]

        axis.scatter(
            sub_df["z_depth"],
            sub_df[metric],
            s=40,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.4,
            label=visit,
        )

    z_means = df.groupby("z_depth")[metric].mean().reset_index().sort_values("z_depth")

    axis.plot(
        z_means["z_depth"],
        z_means[metric],
        color="black",
        linewidth=1.8,
        linestyle="--",
        alpha=0.8,
        label="Mean by depth",
    )

    axis.set_xlabel("Imaging depth (z)")
    axis.set_ylabel(metric_title(metric))
    axis.set_title(f"{metric_title(metric)} vs imaging depth")
    axis.grid(alpha=0.18)

    if metric in ["object_precision", "precision"]:
        axis.set_ylim(0, 1.05)

    if metric == "false_positive_percentage":
        axis.set_ylim(0, 100)

    axis.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_boxplot_by_depth(df, metric, output_path):
    groups = ["superficial", "mid", "deep"]

    data = [
        df.loc[df["depth_group"] == group, metric].dropna().values for group in groups
    ]

    fig, axis = plt.subplots(figsize=(5.2, 4.5), dpi=FIG_DPI)

    boxplot = axis.boxplot(
        data,
        labels=groups,
        patch_artist=True,
        widths=0.6,
        showfliers=True,
    )

    for patch in boxplot["boxes"]:
        patch.set_facecolor("#D9D9D9")
        patch.set_edgecolor("black")

    rng = np.random.default_rng(0)

    for index, values in enumerate(data):
        x_values = (index + 1) + rng.normal(0, 0.04, size=len(values))

        axis.scatter(
            x_values,
            values,
            s=18,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )

    axis.set_ylabel(metric_title(metric))
    axis.set_title(f"{metric_title(metric)} by depth group")
    axis.grid(axis="y", alpha=0.18)

    if metric in ["object_precision", "precision"]:
        axis.set_ylim(0, 1.05)

    if metric == "false_positive_percentage":
        axis.set_ylim(0, 100)

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_summary_panel(df, output_path):
    metrics = [
        "object_precision",
        "false_positive_percentage",
    ]

    groups = ["superficial", "mid", "deep"]

    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(8.5, 4.2),
        dpi=FIG_DPI,
    )

    for axis, metric in zip(axes, metrics):
        means = []

        for group in groups:
            values = df.loc[df["depth_group"] == group, metric].dropna().values
            means.append(np.nanmean(values) if len(values) > 0 else np.nan)

        bars = axis.bar(
            groups,
            means,
            alpha=0.85,
            width=0.7,
        )

        for bar in bars:
            bar.set_edgecolor("black")
            bar.set_linewidth(0.8)

        axis.set_title(metric_title(metric))
        axis.grid(axis="y", alpha=0.18)

        if metric == "object_precision":
            axis.set_ylim(0, 1.05)

        if metric == "false_positive_percentage":
            axis.set_ylim(0, 100)

    fig.suptitle(
        "Segmentation performance by imaging depth",
        fontsize=12,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# =========================================================
# MAIN
# =========================================================


def main():
    set_style()

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"No encontré:\n{CSV_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    required_columns = [
        "visit",
        "mosaic",
        "object_precision",
        "false_positive_percentage",
        "n_gt_objects",
        "n_pred_objects",
    ]

    validate_columns(df, required_columns)

    df["z_depth"] = df["mosaic"].apply(extract_z_depth)
    df["depth_group"] = df["z_depth"].apply(assign_depth_group)

    out_csv = OUTPUT_DIR / "tile_metrics_with_depth.csv"
    df.to_csv(out_csv, index=False)

    scatter_metrics = [
        "object_precision",
        "false_positive_percentage",
        "n_gt_objects",
        "n_pred_objects",
    ]

    for metric in scatter_metrics:
        save_scatter(
            df,
            metric,
            OUTPUT_DIR / f"scatter_{metric}_vs_depth.png",
        )

    for metric in [
        "object_precision",
        "false_positive_percentage",
    ]:
        save_boxplot_by_depth(
            df,
            metric,
            OUTPUT_DIR / f"boxplot_{metric}_by_depth_group.png",
        )

    save_summary_panel(
        df,
        OUTPUT_DIR / "summary_panel_by_depth.png",
    )

    summary = df.groupby("depth_group")[
        [
            "object_precision",
            "false_positive_percentage",
            "n_gt_objects",
            "n_pred_objects",
        ]
    ].agg(["mean", "std", "median", "count"])

    summary_csv = OUTPUT_DIR / "depth_group_summary.csv"
    summary.to_csv(summary_csv)

    print("\n[DONE]")
    print(f"Input CSV:\n{CSV_PATH}")
    print(f"Updated CSV:\n{out_csv}")
    print(f"Summary:\n{summary_csv}")
    print(f"Results saved in:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
