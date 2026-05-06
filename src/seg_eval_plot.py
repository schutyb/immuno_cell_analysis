#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Plot the key metrics for Unet of PC and final area+phasor segmentation against manual masks.

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================

EVAL_DIR = Path(
    "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449/segmentation_evaluation"
).expanduser()

TILE_CSV = EVAL_DIR / "segmentation_key_metrics_by_tile.csv"

OUTPUT_DIR = EVAL_DIR / "key_metrics_final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_DPI = 600

METHOD_ORDER = [
    "PeterChang_UNet",
    "PeterChang_area_FLIM",
]

METHOD_LABELS = {
    "PeterChang_UNet": "U-Net",
    "PeterChang_area_FLIM": "U-Net + area/FLIM",
}

KEY_METRICS = [
    "precision",
    "object_precision",
    "fp_objects",
    "relative_cell_count_error",
]

METRIC_LABELS = {
    "precision": "Pixel-level precision",
    "object_precision": "Object-level precision",
    "fp_objects": "False-positive objects per tile",
    "relative_cell_count_error": "Relative cell count error",
}


# =========================
# HELPERS
# =========================

def metric_title(metric):
    return METRIC_LABELS.get(metric, metric.replace("_", " "))


def clean_df(df):
    df = df.copy()
    df = df[df["method"].isin(METHOD_ORDER)]
    df["method_label"] = df["method"].map(METHOD_LABELS)
    return df


def build_summary(df):
    summary = (
        df.groupby("method")[KEY_METRICS]
        .agg(["mean", "std", "median", "count"])
        .reindex(METHOD_ORDER)
    )

    rows = []

    for method in METHOD_ORDER:
        if method not in summary.index:
            continue

        row = {
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
        }

        for metric in KEY_METRICS:
            if metric not in df.columns:
                continue

            row[f"{metric}_mean"] = summary.loc[method, (metric, "mean")]
            row[f"{metric}_std"] = summary.loc[method, (metric, "std")]
            row[f"{metric}_median"] = summary.loc[method, (metric, "median")]
            row[f"{metric}_n"] = summary.loc[method, (metric, "count")]

        rows.append(row)

    return pd.DataFrame(rows)


def add_change_rows(summary_df):
    """
    Adds absolute and fold-change rows comparing U-Net + area/FLIM vs U-Net.
    """
    if len(summary_df) < 2:
        return pd.DataFrame()

    base = summary_df[summary_df["method"] == METHOD_ORDER[0]].iloc[0]
    filt = summary_df[summary_df["method"] == METHOD_ORDER[1]].iloc[0]

    rows = []

    for metric in KEY_METRICS:
        base_mean = base[f"{metric}_mean"]
        filt_mean = filt[f"{metric}_mean"]

        abs_change = filt_mean - base_mean

        if base_mean != 0 and np.isfinite(base_mean):
            fold_change = filt_mean / base_mean
            percent_change = 100 * (filt_mean - base_mean) / base_mean
        else:
            fold_change = np.nan
            percent_change = np.nan

        rows.append(
            {
                "metric": metric,
                "metric_label": metric_title(metric),
                "unet_mean": base_mean,
                "unet_area_flim_mean": filt_mean,
                "absolute_change": abs_change,
                "fold_change": fold_change,
                "percent_change": percent_change,
            }
        )

    return pd.DataFrame(rows)


def save_key_boxplot_panel(df, out_path):
    metrics = [m for m in KEY_METRICS if m in df.columns]

    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(5.0 * len(metrics), 5.2),
    )

    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        data = []
        labels = []

        for method in METHOD_ORDER:
            values = df.loc[df["method"] == method, metric].dropna().values
            if len(values) == 0:
                continue

            data.append(values)
            labels.append(METHOD_LABELS.get(method, method))

        ax.boxplot(
            data,
            labels=labels,
            showmeans=True,
            meanprops={
                "marker": "o",
                "markerfacecolor": "black",
                "markeredgecolor": "black",
                "markersize": 5,
            },
        )

        ax.set_title(metric_title(metric), fontsize=11)
        ax.grid(axis="y", alpha=0.25)

        if metric in ["precision", "object_precision"]:
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Score")
        elif metric == "relative_cell_count_error":
            ax.set_ylabel("Relative error")
        else:
            ax.set_ylabel("Objects / tile")

    fig.suptitle(
        "Key segmentation performance metrics",
        fontsize=15,
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


def save_explanation_txt(summary_df, changes_df, out_path):
    lines = []

    lines.append("Key segmentation performance metrics\n")
    lines.append("=" * 45 + "\n\n")

    lines.append(
        "This analysis focuses only on the metrics most relevant for this project. "
        "The goal is not perfect boundary reconstruction, but biologically specific "
        "immune-cell detection and reliable cell quantification.\n\n"
    )

    lines.append("Methods compared:\n")
    lines.append("- U-Net: Peter Chang U-Net segmentation output.\n")
    lines.append(
        "- U-Net + area/FLIM: U-Net candidate masks refined using area filtering "
        "and ROI-level FLIM phasor/lifetime filtering.\n\n"
    )

    metric_explanations = {
        "precision": (
            "Pixel-level precision measures the fraction of all pixels predicted "
            "as cell that are also labeled as cell in the manual expert mask. "
            "It is calculated as TP / (TP + FP), where TP are correctly predicted "
            "cell pixels and FP are predicted cell pixels outside the manual mask. "
            "Higher values indicate fewer non-cell pixels incorrectly included "
            "in the segmentation."
        ),
        "object_precision": (
            "Object-level precision measures the fraction of detected objects "
            "that correspond to true manually annotated cells. Predicted connected "
            "components are matched to manual cell objects using object overlap. "
            "It is calculated as TP_objects / (TP_objects + FP_objects). "
            "This is highly relevant here because the goal is to identify real "
            "cellular structures rather than perfectly reproduce object boundaries."
        ),
        "fp_objects": (
            "False-positive objects per tile measures the number of predicted "
            "objects that do not sufficiently overlap with any manual cell object. "
            "These objects typically correspond to non-cellular structures, noise, "
            "elastin, melanin, or other false detections. Lower values indicate "
            "better biological specificity."
        ),
        "relative_cell_count_error": (
            "Relative cell count error measures how far the predicted cell count "
            "is from the manual count. It is calculated as |N_pred - N_GT| / N_GT, "
            "where N_pred is the number of predicted objects and N_GT is the number "
            "of manually annotated cells. Lower values indicate more accurate "
            "cell quantification."
        ),
    }

    for metric in KEY_METRICS:
        lines.append(f"{metric_title(metric)}\n")
        lines.append("-" * len(metric_title(metric)) + "\n")
        lines.append(metric_explanations[metric] + "\n\n")

        for _, row in summary_df.iterrows():
            method_label = row["method_label"]
            mean = row[f"{metric}_mean"]
            std = row[f"{metric}_std"]
            median = row[f"{metric}_median"]
            n = int(row[f"{metric}_n"])

            lines.append(
                f"{method_label}: mean = {mean:.4f}, "
                f"std = {std:.4f}, median = {median:.4f}, n = {n}\n"
            )

        change_row = changes_df[changes_df["metric"] == metric]
        if len(change_row) > 0:
            change_row = change_row.iloc[0]
            lines.append(
                f"Change after area/FLIM filtering: "
                f"{change_row['absolute_change']:+.4f} "
                f"({change_row['percent_change']:+.1f}%).\n"
            )

            if metric in ["precision", "object_precision"]:
                lines.append(
                    "Interpretation: an increase indicates improved specificity "
                    "and fewer false-positive cell detections.\n\n"
                )
            else:
                lines.append(
                    "Interpretation: a decrease indicates improved performance, "
                    "with fewer false detections or more accurate cell counts.\n\n"
                )

    lines.append("Overall interpretation\n")
    lines.append("-" * 22 + "\n")
    lines.append(
        "The key expected outcome is that U-Net + area/FLIM filtering improves "
        "precision and object-level precision while reducing false-positive objects "
        "and relative cell count error. These metrics are more informative than "
        "Dice/IoU for this biological task because the downstream goal is accurate "
        "immune-cell detection and quantification, not perfect boundary overlap.\n"
    )

    out_path.write_text("".join(lines))


# =========================
# MAIN
# =========================

def main():
    if not TILE_CSV.exists():
        raise FileNotFoundError(f"No encontré el CSV:\n{TILE_CSV}")

    df = pd.read_csv(TILE_CSV)
    df = clean_df(df)

    missing = [m for m in KEY_METRICS if m not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")

    print(f"Loaded: {TILE_CSV}")
    print(f"Rows used: {len(df)}")
    print(f"Output dir: {OUTPUT_DIR}")

    # Save only relevant tile-level values
    key_tile_csv = OUTPUT_DIR / "key_metrics_by_tile.csv"
    keep_cols = [
        c for c in [
            "visit",
            "mosaic",
            "tile",
            "method",
            "method_label",
            *KEY_METRICS,
        ]
        if c in df.columns
    ]

    df[keep_cols].to_csv(key_tile_csv, index=False)

    # Summary CSV
    summary_df = build_summary(df)
    changes_df = add_change_rows(summary_df)

    summary_csv = OUTPUT_DIR / "key_metrics_summary.csv"
    changes_csv = OUTPUT_DIR / "key_metrics_changes.csv"

    summary_df.to_csv(summary_csv, index=False)
    changes_df.to_csv(changes_csv, index=False)

    # Figure
    fig_path = OUTPUT_DIR / "key_metrics_boxplots.png"
    save_key_boxplot_panel(df, fig_path)

    # TXT explanation
    txt_path = OUTPUT_DIR / "key_metrics_explanation.txt"
    save_explanation_txt(summary_df, changes_df, txt_path)

    print("\nDone.")
    print(f"Tile-level key metrics:\n{key_tile_csv}")
    print(f"Summary:\n{summary_csv}")
    print(f"Changes:\n{changes_csv}")
    print(f"Figure:\n{fig_path}")
    print(f"Explanation:\n{txt_path}")


if __name__ == "__main__":
    main()