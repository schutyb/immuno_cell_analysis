#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phasorpy.plot import plot_phasor


# ============================================================
# CONFIG
# ============================================================

PATIENT_DIR = Path("/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449")
ANALYSIS_DIR = PATIENT_DIR / "analysis"

RAW_CSV = ANALYSIS_DIR / "roi_gmm_raw_phasor" / "cluster_centers_raw_phasor.csv"
ELASTIN_ONLY_CSV = ANALYSIS_DIR / "roi_gmm_comparison_all_visits" / "cluster_centers_elastin_only_pipeline.csv"
COUMARIN_CSV = ANALYSIS_DIR / "roi_gmm_comparison_all_visits" / "cluster_centers_coumarin_pipeline.csv"

OUTPUT_DIR = ANALYSIS_DIR / "cluster_center_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PHASOR_FREQUENCY = 80.0
SHOW_PLOTS = True

CLASS_COLORS = {
    "elastin": "green",
    "cells": "red",
    "melanin": "saddlebrown",
}

PIPELINE_STYLES = {
    "raw": {
        "marker": "o",
        "size": 120,
        "filled": False,
        "label": "raw",
    },
    "elastin_only": {
        "marker": "X",
        "size": 160,
        "filled": True,
        "label": "uncalibrated + elastin correction",
    },
    "coumarin": {
        "marker": "s",
        "size": 120,
        "filled": True,
        "label": "coumarin pipeline",
    },
}


# ============================================================
# HELPERS
# ============================================================

def load_cluster_csv(csv_path: Path, pipeline_name: str) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"visit", "bio_label", "g_mean", "s_mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} missing columns: {sorted(missing)}")

    df = df.copy()
    df["pipeline"] = pipeline_name
    return df


def unique_visits_in_order(*dfs: pd.DataFrame) -> list[str]:
    visits = []
    for df in dfs:
        for v in df["visit"].tolist():
            if v not in visits:
                visits.append(v)
    return sorted(visits)


def plot_visit_centers(
    df_visit: pd.DataFrame,
    visit: str,
    outpath: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))

    plot_phasor(
        np.array([0.5]),
        np.array([0.0]),
        style="plot",
        marker="",
        linestyle="",
        frequency=PHASOR_FREQUENCY,
        ax=ax,
        title=f"Cluster centers by pipeline - {visit}",
        show=False,
    )

    for _, row in df_visit.iterrows():
        bio = row["bio_label"]
        pipeline = row["pipeline"]

        color = CLASS_COLORS.get(bio, "black")
        style = PIPELINE_STYLES[pipeline]

        facecolor = color if style["filled"] else "none"

        ax.scatter(
            row["g_mean"],
            row["s_mean"],
            s=style["size"],
            marker=style["marker"],
            facecolors=facecolor,
            edgecolors=color,
            linewidths=1.8,
            zorder=10,
            label=f"{bio} | {style['label']}",
        )

    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), fontsize=8, loc="best")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 0.7)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def plot_class_trajectory(
    df_all: pd.DataFrame,
    bio_label: str,
    outpath: Path,
) -> None:
    df_class = df_all[df_all["bio_label"] == bio_label].copy()
    if df_class.empty:
        print(f"[WARN] No rows for class: {bio_label}")
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    plot_phasor(
        np.array([0.5]),
        np.array([0.0]),
        style="plot",
        marker="",
        linestyle="",
        frequency=PHASOR_FREQUENCY,
        ax=ax,
        title=f"Cluster center trajectories - {bio_label}",
        show=False,
    )

    color = CLASS_COLORS.get(bio_label, "black")

    visits = sorted(df_class["visit"].unique())
    pipeline_order = ["raw", "elastin_only", "coumarin"]

    for visit in visits:
        dv = df_class[df_class["visit"] == visit].copy()
        if dv.empty:
            continue

        # plot points
        points = []
        for pipeline in pipeline_order:
            row = dv[dv["pipeline"] == pipeline]
            if row.empty:
                continue
            row = row.iloc[0]

            style = PIPELINE_STYLES[pipeline]
            facecolor = color if style["filled"] else "none"

            ax.scatter(
                row["g_mean"],
                row["s_mean"],
                s=style["size"],
                marker=style["marker"],
                facecolors=facecolor,
                edgecolors=color,
                linewidths=1.8,
                zorder=10,
                label=f"{visit} | {style['label']}",
            )
            points.append((row["g_mean"], row["s_mean"]))

        # connect raw -> elastin_only -> coumarin
        if len(points) >= 2:
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            ax.plot(x, y, color=color, linewidth=1.2, alpha=0.8)

    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), fontsize=8, loc="best")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 0.7)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def save_combined_csv(df_all: pd.DataFrame, outpath: Path) -> None:
    cols = ["visit", "bio_label", "pipeline", "g_mean", "s_mean"]
    extra = [c for c in df_all.columns if c not in cols]
    df_all[cols + extra].to_csv(outpath, index=False)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    df_raw = load_cluster_csv(RAW_CSV, "raw")
    df_elastin = load_cluster_csv(ELASTIN_ONLY_CSV, "elastin_only")
    df_coumarin = load_cluster_csv(COUMARIN_CSV, "coumarin")

    df_all = pd.concat([df_raw, df_elastin, df_coumarin], ignore_index=True)
    save_combined_csv(df_all, OUTPUT_DIR / "cluster_centers_all_pipelines.csv")

    visits = unique_visits_in_order(df_raw, df_elastin, df_coumarin)

    for visit in visits:
        df_visit = df_all[df_all["visit"] == visit].copy()
        if df_visit.empty:
            continue

        plot_visit_centers(
            df_visit,
            visit,
            OUTPUT_DIR / f"{visit}_cluster_centers_all_pipelines.png",
        )
        print(f"[OK] Saved per-visit comparison: {visit}")

    for bio_label in ["elastin", "cells", "melanin"]:
        plot_class_trajectory(
            df_all,
            bio_label,
            OUTPUT_DIR / f"{bio_label}_cluster_center_trajectories.png",
        )
        print(f"[OK] Saved class trajectory: {bio_label}")

    print("[DONE]")
    print(f"Saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()