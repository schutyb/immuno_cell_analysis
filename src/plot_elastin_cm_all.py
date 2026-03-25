#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

CLUSTER_CSV = ANALYSIS_DIR / "cluster_centers_all_three_types_by_visit.csv"

OUTPUT_DIR = ANALYSIS_DIR / "elastin_cm_only"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUTPUT_DIR / "elastin_cluster_centers_only.png"

PHASOR_FREQUENCY = 80.0
SHOW_PLOT = True

# color by visit
VISIT_COLORS = {
    "visit01": "red",
    "visit02": "blue",
    "visit03": "green",
    "visit04": "orange",
}

# style by phasor type
PHASOR_TYPE_STYLE = {
    "uncalibrated": {
        "marker": "o",
        "size": 220,
        "legend": "uncalibrated",
    },
    "uncalibrated_elastin_corr": {
        "marker": "x",
        "size": 260,
        "legend": "uncalibrated + elastin",
    },
    "coumarin_calibrated": {
        "marker": "*",
        "size": 260,
        "legend": "coumarin calibrated",
    },
}


# ============================================================
# HELPERS
# ============================================================

def load_cluster_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"visit", "phasor_type", "bio_label", "g_mean", "s_mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} missing columns: {sorted(missing)}")

    return df.copy()


def short_visit_label(visit: str) -> str:
    visit = str(visit).lower()
    if visit.startswith("visit"):
        try:
            n = int(visit.replace("visit", ""))
            return f"v{n}"
        except ValueError:
            return visit
    return visit


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    df = load_cluster_csv(CLUSTER_CSV)

    # only elastin
    df = df[df["bio_label"].str.lower() == "elastin"].copy()
    if df.empty:
        raise RuntimeError("No elastin rows found in the CSV.")

    print("\nElastin cluster centers:")
    print(df[["visit", "phasor_type", "g_mean", "s_mean"]])

    fig, ax = plt.subplots(figsize=(9, 7))

    plot_phasor(
        np.array([0.5]),
        np.array([0.0]),
        style="plot",
        marker="",
        linestyle="",
        frequency=PHASOR_FREQUENCY,
        ax=ax,
        title="Elastin cluster centers across visits and phasor types",
        show=False,
    )

    for _, row in df.iterrows():
        visit = str(row["visit"]).lower()
        phasor_type = row["phasor_type"]

        color = VISIT_COLORS.get(visit, "black")
        style = PHASOR_TYPE_STYLE.get(phasor_type)
        if style is None:
            continue

        g = float(row["g_mean"])
        s = float(row["s_mean"])
        marker = style["marker"]

        # avoid warning for x marker
        if marker in {"x", "+"}:
            ax.scatter(
                g,
                s,
                s=style["size"],
                marker=marker,
                c=color,
                linewidths=2.0,
                zorder=10,
                label=f"{short_visit_label(visit)} | {style['legend']}",
            )
        else:
            ax.scatter(
                g,
                s,
                s=style["size"],
                marker=marker,
                facecolors="none",
                edgecolors=color,
                linewidths=1.8,
                zorder=10,
                label=f"{short_visit_label(visit)} | {style['legend']}",
            )

    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), fontsize=9, loc="best")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 0.7)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")

    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)

    print(f"\n[DONE] Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()