#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

from skimage.measure import label, regionprops
from sklearn.mixture import GaussianMixture

from phasorpy.plot import plot_phasor


# ============================================================
# CONFIG
# ============================================================

PATIENT_DIR = Path("/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449")

OUTPUT_DIR = PATIENT_DIR / "analysis" / "roi_gmm_comparison_all_visits"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PHASOR_G_IDX = 1
PHASOR_S_IDX = 2
MIN_ROI_AREA = 1
PHASOR_FREQUENCY = 80.0
SHOW_PLOTS = False

COLORS = {
    "elastin": "green",
    "cells": "red",
    "melanin": "saddlebrown",
}

FILTER_PHASOR_RANGE = False
G_MIN, G_MAX = -0.2, 1.2
S_MIN, S_MAX = -0.2, 1.2


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Case:
    patient: str
    visit: str
    mosaic_name: str
    coumarin_path: Path
    elastin_only_path: Path
    mask_path: Path


# ============================================================
# HELPERS
# ============================================================

def infer_patient_visit_from_path(path: Path) -> Tuple[str, str]:
    path_str = str(path)

    patient_match = re.search(r"(p\d+)", path_str, re.IGNORECASE)
    visit_match = re.search(r"(visit[_-]?\d+)", path_str, re.IGNORECASE)

    patient = patient_match.group(1) if patient_match else "unknown_patient"
    visit = visit_match.group(1) if visit_match else "unknown_visit"
    return patient, visit


def n_components_for_visit(visit: str) -> int:
    return 2 if visit.lower() == "visit04" else 3


# ============================================================
# SEARCH
# ============================================================

def find_mask_in_new_folder(mosaic_dir: Path) -> Optional[Path]:
    candidates = [
        mosaic_dir / "_new" / "instance_mask_filtered.tif",
        mosaic_dir / "_new" / "instance_mask_filtered.tiff",
        mosaic_dir / "_new" / "instance_mask_filtered.png",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def collect_cases(patient_dir: Path) -> List[Case]:
    """
    Look for mosaics that have:
      - _new/phasor_corr.tif
      - phasor_uncalibrated_elastin_corr.tif
      - _new/instance_mask_filtered.*
    """
    cases: List[Case] = []

    for coumarin_path in patient_dir.rglob("phasor_corr.tif"):
        # only use the coumarin-corrected phasor inside _new
        if coumarin_path.parent.name != "_new":
            continue

        mosaic_dir = coumarin_path.parent.parent
        elastin_only_path = mosaic_dir / "phasor_uncalibrated_elastin_corr.tif"
        mask_path = find_mask_in_new_folder(mosaic_dir)

        if not elastin_only_path.exists():
            print(f"[SKIP] Missing elastin-only phasor: {elastin_only_path}")
            continue

        if mask_path is None:
            print(f"[SKIP] Missing instance_mask_filtered in: {mosaic_dir / '_new'}")
            continue

        patient, visit = infer_patient_visit_from_path(mosaic_dir)
        cases.append(
            Case(
                patient=patient,
                visit=visit,
                mosaic_name=mosaic_dir.name,
                coumarin_path=coumarin_path,
                elastin_only_path=elastin_only_path,
                mask_path=mask_path,
            )
        )

    return sorted(cases, key=lambda c: (c.patient, c.visit, c.mosaic_name))


# ============================================================
# READERS
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
            raise ValueError(f"Unsupported mask shape: {mask.shape} in {mask_path}")

    if mask.dtype == bool:
        return label(mask).astype(np.int32)

    unique_vals = np.unique(mask)
    if set(unique_vals.tolist()).issubset({0, 1, 255}):
        return label(mask > 0).astype(np.int32)

    return mask.astype(np.int32)


def read_phasor_gs(phasor_path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = tifffile.imread(phasor_path)
    arr = np.asarray(arr).squeeze()

    if arr.ndim != 3:
        raise ValueError(f"Unexpected phasor shape: {arr.shape} in {phasor_path}")

    g = arr[PHASOR_G_IDX].astype(np.float64)
    s = arr[PHASOR_S_IDX].astype(np.float64)
    return g, s


# ============================================================
# ROI TABLE
# ============================================================

def roi_table_from_phasor(
    phasor_path: Path,
    mask_path: Path,
    source_name: str,
    patient: str,
    visit: str,
    mosaic_name: str,
) -> pd.DataFrame:
    g, s = read_phasor_gs(phasor_path)
    labels = read_mask(mask_path)

    if g.shape != labels.shape or s.shape != labels.shape:
        raise ValueError(
            f"Shape mismatch: G={g.shape}, S={s.shape}, mask={labels.shape}"
        )

    rows = []
    props = regionprops(labels)

    for prop in props:
        if prop.area < MIN_ROI_AREA:
            continue

        rr = prop.coords[:, 0]
        cc = prop.coords[:, 1]

        gvals = g[rr, cc]
        svals = s[rr, cc]

        valid = np.isfinite(gvals) & np.isfinite(svals)

        if FILTER_PHASOR_RANGE:
            valid &= (
                (gvals >= G_MIN) & (gvals <= G_MAX) &
                (svals >= S_MIN) & (svals <= S_MAX)
            )

        if valid.sum() == 0:
            continue

        gvals = gvals[valid]
        svals = svals[valid]

        rows.append({
            "patient": patient,
            "visit": visit,
            "mosaic_name": mosaic_name,
            "source": source_name,
            "phasor_path": str(phasor_path),
            "mask_path": str(mask_path),
            "roi_label": int(prop.label),
            "area_px": int(prop.area),
            "centroid_row": float(prop.centroid[0]),
            "centroid_col": float(prop.centroid[1]),
            "g_mean": float(np.mean(gvals)),
            "s_mean": float(np.mean(svals)),
            "g_std_within_roi": float(np.std(gvals, ddof=0)),
            "s_std_within_roi": float(np.std(svals, ddof=0)),
            "n_valid_pixels": int(valid.sum()),
        })

    return pd.DataFrame(rows)


# ============================================================
# GMM + LABELS
# ============================================================

def compute_phase(g: np.ndarray, s: np.ndarray) -> np.ndarray:
    return np.arctan2(s, g)


def assign_biological_labels_by_phase(
    df_roi: pd.DataFrame,
    n_components: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = df_roi[["g_mean", "s_mean"]].to_numpy()

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=0,
        n_init=10,
    )
    gmm.fit(X)

    pred = gmm.predict(X)

    df_out = df_roi.copy()
    df_out["gmm_cluster"] = pred
    df_out["phase"] = compute_phase(df_out["g_mean"].to_numpy(), df_out["s_mean"].to_numpy())

    cluster_rows = []
    for k in sorted(np.unique(pred)):
        d = df_out[df_out["gmm_cluster"] == k]
        cluster_rows.append({
            "cluster": int(k),
            "phase_mean": float(d["phase"].mean()),
            "g_mean": float(d["g_mean"].mean()),
            "s_mean": float(d["s_mean"].mean()),
            "g_std": float(d["g_mean"].std(ddof=0)),
            "s_std": float(d["s_mean"].std(ddof=0)),
            "n_rois": int(len(d)),
        })

    cluster_df = pd.DataFrame(cluster_rows).sort_values("phase_mean").reset_index(drop=True)

    if len(cluster_df) == 3:
        cluster_df.loc[0, "bio_label"] = "melanin"
        cluster_df.loc[1, "bio_label"] = "cells"
        cluster_df.loc[2, "bio_label"] = "elastin"
    elif len(cluster_df) == 2:
        cluster_df.loc[0, "bio_label"] = "cells"
        cluster_df.loc[1, "bio_label"] = "elastin"
    else:
        raise ValueError(f"Expected 2 or 3 clusters, got {len(cluster_df)}")

    cluster_to_bio = dict(zip(cluster_df["cluster"], cluster_df["bio_label"]))
    df_out["bio_label"] = df_out["gmm_cluster"].map(cluster_to_bio)

    return df_out, cluster_df


# ============================================================
# PLOTTING
# ============================================================

def plot_roi_gmm(
    df_roi: pd.DataFrame,
    cluster_df: pd.DataFrame,
    title: str,
    outpath: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    plotted_any = False

    for bio_label in ["elastin", "cells", "melanin"]:
        d = df_roi[df_roi["bio_label"] == bio_label]
        if d.empty:
            continue

        plot_phasor(
            d["g_mean"].to_numpy(),
            d["s_mean"].to_numpy(),
            style="plot",
            marker=".",
            linestyle="",
            color=COLORS[bio_label],
            label=bio_label,
            frequency=PHASOR_FREQUENCY,
            ax=ax,
            title=title,
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
            frequency=PHASOR_FREQUENCY,
            ax=ax,
            title=title,
            show=False,
        )

    for _, row in cluster_df.iterrows():
        ax.scatter(
            row["g_mean"],
            row["s_mean"],
            s=150,
            c=COLORS[row["bio_label"]],
            edgecolors="black",
            linewidths=1.0,
            marker="X",
            zorder=10,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def plot_cluster_center_overlay(
    cluster_df_coumarin: pd.DataFrame,
    cluster_df_elastin: pd.DataFrame,
    title: str,
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
        title=title,
        show=False,
    )

    for _, row in cluster_df_coumarin.iterrows():
        ax.scatter(
            row["g_mean"],
            row["s_mean"],
            s=130,
            c=COLORS[row["bio_label"]],
            edgecolors="black",
            linewidths=1.0,
            marker="o",
            label=f"{row['bio_label']} (coumarin)",
            zorder=10,
        )

    for _, row in cluster_df_elastin.iterrows():
        ax.scatter(
            row["g_mean"],
            row["s_mean"],
            s=160,
            c=COLORS[row["bio_label"]],
            edgecolors="black",
            linewidths=1.0,
            marker="X",
            label=f"{row['bio_label']} (elastin-only)",
            zorder=11,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 0.7)

    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), fontsize=9)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    cases = collect_cases(PATIENT_DIR)

    if not cases:
        raise RuntimeError("No valid mosaics found with both phasor pipelines and instance mask.")

    print(f"[INFO] Found {len(cases)} mosaics")
    for c in cases:
        print(f"  - {c.visit} | {c.mosaic_name}")

    all_coumarin = []
    all_elastin = []

    for case in cases:
        try:
            df_c = roi_table_from_phasor(
                case.coumarin_path,
                case.mask_path,
                source_name="coumarin_pipeline",
                patient=case.patient,
                visit=case.visit,
                mosaic_name=case.mosaic_name,
            )
            df_e = roi_table_from_phasor(
                case.elastin_only_path,
                case.mask_path,
                source_name="elastin_only_pipeline",
                patient=case.patient,
                visit=case.visit,
                mosaic_name=case.mosaic_name,
            )

            if len(df_c) == 0 or len(df_e) == 0:
                print(f"[WARN] Empty ROI table in {case.visit} | {case.mosaic_name}")
                continue

            all_coumarin.append(df_c)
            all_elastin.append(df_e)

            print(
                f"[OK] {case.visit} | {case.mosaic_name} | "
                f"coumarin ROIs={len(df_c)} | elastin-only ROIs={len(df_e)}"
            )

        except Exception as e:
            print(f"[ERROR] {case.visit} | {case.mosaic_name}: {e}")

    if not all_coumarin or not all_elastin:
        raise RuntimeError("No ROI tables could be generated.")

    df_all_coumarin = pd.concat(all_coumarin, ignore_index=True)
    df_all_elastin = pd.concat(all_elastin, ignore_index=True)

    df_all_coumarin.to_csv(OUTPUT_DIR / "roi_all_coumarin_pipeline.csv", index=False)
    df_all_elastin.to_csv(OUTPUT_DIR / "roi_all_elastin_only_pipeline.csv", index=False)

    cluster_rows_c = []
    cluster_rows_e = []

    all_labeled_c = []
    all_labeled_e = []

    for visit in sorted(df_all_coumarin["visit"].unique()):
        df_visit_c = df_all_coumarin[df_all_coumarin["visit"] == visit].copy()
        df_visit_e = df_all_elastin[df_all_elastin["visit"] == visit].copy()

        if len(df_visit_c) < 2 or len(df_visit_e) < 2:
            print(f"[WARN] Too few ROIs in {visit}, skipping GMM")
            continue

        n_components = n_components_for_visit(visit)

        df_visit_c_lab, cluster_c = assign_biological_labels_by_phase(
            df_visit_c, n_components
        )
        df_visit_e_lab, cluster_e = assign_biological_labels_by_phase(
            df_visit_e, n_components
        )

        cluster_c["visit"] = visit
        cluster_e["visit"] = visit

        all_labeled_c.append(df_visit_c_lab)
        all_labeled_e.append(df_visit_e_lab)

        cluster_rows_c.append(cluster_c)
        cluster_rows_e.append(cluster_e)

        # save per-visit plots
        plot_roi_gmm(
            df_visit_c_lab,
            cluster_c,
            title=f"ROI phasor GMM - Coumarin pipeline - {visit}",
            outpath=OUTPUT_DIR / f"{visit}_roi_phasor_gmm_coumarin_pipeline.png",
        )

        plot_roi_gmm(
            df_visit_e_lab,
            cluster_e,
            title=f"ROI phasor GMM - Uncalibrated + elastin correction - {visit}",
            outpath=OUTPUT_DIR / f"{visit}_roi_phasor_gmm_elastin_only_pipeline.png",
        )

        plot_cluster_center_overlay(
            cluster_c,
            cluster_e,
            title=f"Cluster centers: Coumarin vs Elastin-only - {visit}",
            outpath=OUTPUT_DIR / f"{visit}_roi_phasor_gmm_cluster_center_overlay.png",
        )

        print(f"[OK] Saved plots for {visit}")

    if all_labeled_c:
        pd.concat(all_labeled_c, ignore_index=True).to_csv(
            OUTPUT_DIR / "roi_labeled_coumarin_pipeline.csv",
            index=False,
        )

    if all_labeled_e:
        pd.concat(all_labeled_e, ignore_index=True).to_csv(
            OUTPUT_DIR / "roi_labeled_elastin_only_pipeline.csv",
            index=False,
        )

    if cluster_rows_c:
        pd.concat(cluster_rows_c, ignore_index=True).to_csv(
            OUTPUT_DIR / "cluster_centers_coumarin_pipeline.csv",
            index=False,
        )

    if cluster_rows_e:
        pd.concat(cluster_rows_e, ignore_index=True).to_csv(
            OUTPUT_DIR / "cluster_centers_elastin_only_pipeline.csv",
            index=False,
        )

    print("[DONE]")
    print(f"Saved everything in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()