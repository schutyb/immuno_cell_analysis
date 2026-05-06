#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

FINAL_EVAL_DIR = EVAL_DIR / "final_unet_flim_evaluation"

TILE_CSV = FINAL_EVAL_DIR / "unet_flim_key_metrics_by_tile.csv"

OUTPUT_DIR = FINAL_EVAL_DIR / "plot_options"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_DPI = 600

METHOD_LABEL = "U-Net + FLIM"

KEY_METRICS = [
    "precision",
    "object_precision",
    "fp_objects",
    "relative_cell_count_error",
]

METRIC_LABELS = {
    "precision": "Pixel-level precision",
    "object_precision": "Object-level precision",
    "fp_objects": "False-positive objects / tile",
    "relative_cell_count_error": "Relative cell count error",
}

METRIC_YLABELS = {
    "precision": "Score",
    "object_precision": "Score",
    "fp_objects": "Objects / tile",
    "relative_cell_count_error": "Relative error",
}


# =========================
# HELPERS
# =========================

def metric_title(metric):
    return METRIC_LABELS.get(metric, metric.replace("_", " "))


def load_data():
    if not TILE_CSV.exists():
        raise FileNotFoundError(f"No encontré el CSV:\n{TILE_CSV}")

    df = pd.read_csv(TILE_CSV)

    missing = [m for m in KEY_METRICS if m not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")

    return df


def jitter_x(n, center=0, width=0.08, seed=0):
    rng = np.random.default_rng(seed)
    return center + rng.normal(0, width, size=n)


def save_boxplots(df):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))

    for ax, metric in zip(axes, KEY_METRICS):
        values = df[metric].dropna().values

        ax.boxplot(
            [values],
            labels=[METHOD_LABEL],
            showmeans=True,
            meanprops={
                "marker": "o",
                "markerfacecolor": "black",
                "markeredgecolor": "black",
                "markersize": 5,
            },
        )

        ax.set_title(metric_title(metric))
        ax.set_ylabel(METRIC_YLABELS.get(metric, metric))
        ax.grid(axis="y", alpha=0.25)

        if metric in ["precision", "object_precision"]:
            ax.set_ylim(0, 1.05)

    fig.suptitle("Boxplots", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_boxplots.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def save_violin_strip(df):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))

    for i, (ax, metric) in enumerate(zip(axes, KEY_METRICS)):
        values = df[metric].dropna().values

        parts = ax.violinplot(
            [values],
            positions=[0],
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )

        for pc in parts["bodies"]:
            pc.set_alpha(0.35)

        x = jitter_x(len(values), center=0, width=0.045, seed=i)
        ax.scatter(x, values, s=24, alpha=0.65)

        mean = np.mean(values)
        median = np.median(values)

        ax.scatter([0], [mean], s=70, color="black", zorder=5, label="Mean")
        ax.hlines(
            median,
            -0.15,
            0.15,
            color="black",
            linewidth=2,
            label="Median",
        )

        ax.set_xticks([0])
        ax.set_xticklabels([METHOD_LABEL])
        ax.set_title(metric_title(metric))
        ax.set_ylabel(METRIC_YLABELS.get(metric, metric))
        ax.grid(axis="y", alpha=0.25)

        if metric in ["precision", "object_precision"]:
            ax.set_ylim(0, 1.05)

    fig.suptitle("Violin + individual tiles", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_violin_strip.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def save_bar_points(df):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))

    for i, (ax, metric) in enumerate(zip(axes, KEY_METRICS)):
        values = df[metric].dropna().values

        mean = np.mean(values)
        std = np.std(values, ddof=1)

        ax.bar(
            [0],
            [mean],
            yerr=[std],
            capsize=6,
            alpha=0.55,
            width=0.55,
        )

        x = jitter_x(len(values), center=0, width=0.045, seed=i)
        ax.scatter(x, values, s=24, alpha=0.65)

        ax.set_xticks([0])
        ax.set_xticklabels([METHOD_LABEL])
        ax.set_title(metric_title(metric))
        ax.set_ylabel(METRIC_YLABELS.get(metric, metric))
        ax.grid(axis="y", alpha=0.25)

        ax.text(
            0,
            mean,
            f"{mean:.2f} ± {std:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

        if metric in ["precision", "object_precision"]:
            ax.set_ylim(0, 1.05)

    fig.suptitle("Mean ± SD with individual tiles", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_bar_points_mean_sd.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def save_lollipop(df):
    metrics = KEY_METRICS

    means = [df[m].mean() for m in metrics]
    stds = [df[m].std() for m in metrics]

    labels = [metric_title(m) for m in metrics]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    y = np.arange(len(metrics))

    ax.hlines(y, 0, means, linewidth=2, alpha=0.8)
    ax.errorbar(
        means,
        y,
        xerr=stds,
        fmt="o",
        color="black",
        capsize=4,
        markersize=7,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Metric value")
    ax.set_title("Lollipop plot | mean ± SD")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_lollipop_mean_sd.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def save_performance_cards(df):
    summary = {}

    for metric in KEY_METRICS:
        values = df[metric].dropna().values
        summary[metric] = {
            "mean": np.mean(values),
            "std": np.std(values, ddof=1),
            "median": np.median(values),
            "n": len(values),
        }

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))

    for ax, metric in zip(axes, KEY_METRICS):
        ax.axis("off")

        mean = summary[metric]["mean"]
        std = summary[metric]["std"]
        median = summary[metric]["median"]
        n = summary[metric]["n"]

        ax.text(
            0.5,
            0.78,
            metric_title(metric),
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

        ax.text(
            0.5,
            0.50,
            f"{mean:.2f}",
            ha="center",
            va="center",
            fontsize=34,
            fontweight="bold",
        )

        ax.text(
            0.5,
            0.32,
            f"mean ± SD: {mean:.2f} ± {std:.2f}",
            ha="center",
            va="center",
            fontsize=10,
        )

        ax.text(
            0.5,
            0.22,
            f"median: {median:.2f} | n={n}",
            ha="center",
            va="center",
            fontsize=10,
        )

        ax.add_patch(
            plt.Rectangle(
                (0.03, 0.08),
                0.94,
                0.84,
                fill=False,
                linewidth=1.2,
                alpha=0.6,
                transform=ax.transAxes,
            )
        )

    fig.suptitle("Final segmentation performance: U-Net + FLIM", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_performance_cards.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def save_cdf_plots(df):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))

    for ax, metric in zip(axes, KEY_METRICS):
        values = np.sort(df[metric].dropna().values)
        y = np.arange(1, len(values) + 1) / len(values)

        ax.plot(values, y, linewidth=2)
        ax.scatter(values, y, s=12, alpha=0.55)

        ax.set_title(metric_title(metric))
        ax.set_xlabel(METRIC_YLABELS.get(metric, metric))
        ax.set_ylabel("Cumulative fraction of tiles")
        ax.grid(alpha=0.25)

        if metric in ["precision", "object_precision"]:
            ax.set_xlim(0, 1.05)

    fig.suptitle("Cumulative distribution across evaluated tiles", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_cdf_plots.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def save_compact_nature_style(df):
    """
    A clean 2-row panel:
    Top: cards with mean values.
    Bottom: strip/box hybrid plots.
    """
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.6])

    for i, metric in enumerate(KEY_METRICS):
        values = df[metric].dropna().values
        mean = np.mean(values)
        std = np.std(values, ddof=1)

        ax_card = fig.add_subplot(gs[0, i])
        ax_card.axis("off")

        ax_card.text(
            0.5,
            0.72,
            metric_title(metric),
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

        ax_card.text(
            0.5,
            0.42,
            f"{mean:.2f}",
            ha="center",
            va="center",
            fontsize=28,
            fontweight="bold",
        )

        ax_card.text(
            0.5,
            0.18,
            f"± {std:.2f}",
            ha="center",
            va="center",
            fontsize=11,
        )

        ax_card.add_patch(
            plt.Rectangle(
                (0.04, 0.08),
                0.92,
                0.84,
                fill=False,
                linewidth=1.0,
                alpha=0.5,
                transform=ax_card.transAxes,
            )
        )

        ax = fig.add_subplot(gs[1, i])

        ax.boxplot(
            [values],
            positions=[0],
            widths=0.35,
            showfliers=False,
            showmeans=True,
            meanprops={
                "marker": "o",
                "markerfacecolor": "black",
                "markeredgecolor": "black",
                "markersize": 5,
            },
        )

        x = jitter_x(len(values), center=0, width=0.05, seed=i)
        ax.scatter(x, values, s=22, alpha=0.65)

        ax.set_xticks([0])
        ax.set_xticklabels([METHOD_LABEL])
        ax.set_ylabel(METRIC_YLABELS.get(metric, metric))
        ax.grid(axis="y", alpha=0.25)

        if metric in ["precision", "object_precision"]:
            ax.set_ylim(0, 1.05)

    fig.suptitle(
        "Final segmentation validation against manual annotations",
        fontsize=16,
        y=0.995,
    )

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "07_compact_nature_style_panel.png",
        dpi=FIG_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_summary_csv(df):
    rows = []

    for metric in KEY_METRICS:
        values = df[metric].dropna().values

        rows.append(
            {
                "metric": metric,
                "metric_label": metric_title(metric),
                "mean": np.mean(values),
                "std": np.std(values, ddof=1),
                "median": np.median(values),
                "q1": np.percentile(values, 25),
                "q3": np.percentile(values, 75),
                "min": np.min(values),
                "max": np.max(values),
                "n_tiles": len(values),
            }
        )

    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "unet_flim_plot_summary_stats.csv",
        index=False,
    )


# =========================
# MAIN
# =========================

def main():
    df = load_data()

    print(f"Loaded: {TILE_CSV}")
    print(f"Rows: {len(df)}")
    print(f"Saving plot options to: {OUTPUT_DIR}")

    save_summary_csv(df)
    save_boxplots(df)
    save_violin_strip(df)
    save_bar_points(df)
    save_lollipop(df)
    save_performance_cards(df)
    save_cdf_plots(df)
    save_compact_nature_style(df)

    print("\nDone.")
    print(f"Saved all plot options in:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()