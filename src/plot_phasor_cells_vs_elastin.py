#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import tifffile
import imageio.v3 as iio
import matplotlib.pyplot as plt

from skimage.measure import label, regionprops
from sklearn.mixture import GaussianMixture

from phasorpy.plot import plot_phasor


# =========================
# CONFIG
# =========================

ROOT_DIR = Path("/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449")
OUTPUT_DIR = Path("/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATIENT_FILTER = None  # e.g. "p449" or None

# phasor.tif planes:
# 0 intensity
# 1 G first harmonic
# 2 S first harmonic
# 3 G second harmonic
# 4 S second harmonic
PHASOR_G_IDX = 1
PHASOR_S_IDX = 2

MIN_ROI_AREA = 1
SHOW_PLOTS = True
PHASOR_FREQUENCY = 80.0

# plot style
COLORS = {
    "elastin": "green",
    "cells": "red",
    "melanin": "saddlebrown",
}

# optional sanity filter
FILTER_PHASOR_RANGE = False
G_MIN, G_MAX = -0.2, 1.2
S_MIN, S_MAX = -0.2, 1.2


# =========================
# DATA STRUCTURES
# =========================

@dataclass
class Case:
    phasor_path: Path
    mask_path: Path
    patient: str
    visit: str


# =========================
# SEARCH
# =========================

def infer_patient_visit_from_path(path: Path) -> Tuple[str, str]:
    path_str = str(path)

    patient_match = re.search(r"(p\d+)", path_str, re.IGNORECASE)
    visit_match = re.search(r"(visit[_-]?\d+)", path_str, re.IGNORECASE)

    patient = patient_match.group(1) if patient_match else "unknown_patient"
    visit = visit_match.group(1) if visit_match else "unknown_visit"
    return patient, visit


def find_matching_mask(phasor_path: Path) -> Optional[Path]:
    folder = phasor_path.parent
    candidates = [
        folder / "instance_mask_filtered.tif",
        folder / "instance_mask_filtered.tiff",
        folder / "instance_mask_filtered.png",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def collect_cases(root_dir: Path, patient_filter: Optional[str] = None) -> List[Case]:
    cases: List[Case] = []

    for phasor_path in root_dir.rglob("phasor.tif"):
        if patient_filter is not None and patient_filter.lower() not in str(phasor_path).lower():
            continue

        mask_path = find_matching_mask(phasor_path)
        if mask_path is None:
            print(f"[WARN] No encontré máscara para: {phasor_path}")
            continue

        patient, visit = infer_patient_visit_from_path(phasor_path)
        cases.append(
            Case(
                phasor_path=phasor_path,
                mask_path=mask_path,
                patient=patient,
                visit=visit,
            )
        )

    return sorted(cases, key=lambda c: (c.patient, c.visit, str(c.phasor_path)))


# =========================
# READERS
# =========================

def read_mask(mask_path: Path) -> np.ndarray:
    """
    Reads instance mask from tif/png.
    If binary, connected components are labeled.
    If already labeled, uses labels directly.
    """
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
            raise ValueError(f"Máscara con shape no soportado: {mask.shape} en {mask_path}")

    if mask.dtype == bool:
        return label(mask).astype(np.int32)

    unique_vals = np.unique(mask)

    if set(unique_vals.tolist()).issubset({0, 1, 255}):
        binary = mask > 0
        return label(binary).astype(np.int32)

    return mask.astype(np.int32)


def read_phasor_gs(phasor_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads first-harmonic G and S from phasor.tif
    Expected format: (5, Y, X)
    """
    arr = tifffile.imread(phasor_path)
    arr = np.asarray(arr).squeeze()

    if arr.ndim == 3 and arr.shape[0] == 5:
        g = arr[PHASOR_G_IDX].astype(np.float64)
        s = arr[PHASOR_S_IDX].astype(np.float64)
        return g, s

    if np.iscomplexobj(arr):
        return np.real(arr).astype(np.float64), np.imag(arr).astype(np.float64)

    if arr.ndim == 3 and arr.shape[0] == 2:
        return arr[0].astype(np.float64), arr[1].astype(np.float64)

    if arr.ndim == 3 and arr.shape[-1] == 2:
        return arr[..., 0].astype(np.float64), arr[..., 1].astype(np.float64)

    raise ValueError(
        f"No pude interpretar phasor.tif con shape {arr.shape} en {phasor_path}."
    )


# =========================
# ROI -> ONE POINT
# =========================

def roi_table_from_case(case: Case) -> pd.DataFrame:
    """
    One ROI -> one phasor point (g_mean, s_mean)
    """
    g, s = read_phasor_gs(case.phasor_path)
    labels = read_mask(case.mask_path)

    if g.shape != labels.shape or s.shape != labels.shape:
        raise ValueError(
            f"Shape mismatch en {case.phasor_path.name} vs {case.mask_path.name}: "
            f"G={g.shape}, S={s.shape}, mask={labels.shape}"
        )

    rows = []
    props = regionprops(labels)

    for prop in props:
        roi_label = prop.label
        area = prop.area

        if area < MIN_ROI_AREA:
            continue

        coords = prop.coords
        rr = coords[:, 0]
        cc = coords[:, 1]

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
            "patient": case.patient,
            "visit": case.visit,
            "phasor_path": str(case.phasor_path),
            "mask_path": str(case.mask_path),
            "roi_label": int(roi_label),
            "area_px": int(area),
            "centroid_row": float(prop.centroid[0]),
            "centroid_col": float(prop.centroid[1]),
            "g_mean": float(np.mean(gvals)),
            "s_mean": float(np.mean(svals)),
            "g_std_within_roi": float(np.std(gvals, ddof=0)),
            "s_std_within_roi": float(np.std(svals, ddof=0)),
            "n_valid_pixels": int(valid.sum()),
        })

    return pd.DataFrame(rows)


# =========================
# CLUSTER BIOLOGY
# =========================

def compute_phase(g: np.ndarray, s: np.ndarray) -> np.ndarray:
    return np.arctan2(s, g)


def assign_biological_labels_by_phase(
    df_visit: pd.DataFrame,
    gmm: GaussianMixture,
    visit: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sort clusters by mean phase:
      lowest phase    -> melanin   (if k=3)
      intermediate    -> cells
      highest phase   -> elastin

    For visit04: k=2
      lower phase     -> cells
      higher phase    -> elastin
    """
    X = df_visit[["g_mean", "s_mean"]].to_numpy()
    pred = gmm.predict(X)

    df_out = df_visit.copy()
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
    n_clusters = len(cluster_df)

    if n_clusters == 3:
        cluster_df.loc[0, "bio_label"] = "melanin"
        cluster_df.loc[1, "bio_label"] = "cells"
        cluster_df.loc[2, "bio_label"] = "elastin"
    elif n_clusters == 2:
        cluster_df.loc[0, "bio_label"] = "cells"
        cluster_df.loc[1, "bio_label"] = "elastin"
    else:
        raise ValueError(f"Esperaba 2 o 3 clusters, pero encontré {n_clusters} en {visit}")

    cluster_to_bio = dict(zip(cluster_df["cluster"], cluster_df["bio_label"]))
    df_out["bio_label"] = df_out["gmm_cluster"].map(cluster_to_bio)

    cluster_df["visit"] = visit
    cluster_df = cluster_df[
        ["visit", "cluster", "bio_label", "phase_mean", "g_mean", "s_mean", "g_std", "s_std", "n_rois"]
    ]

    return df_out, cluster_df


# =========================
# PLOTTING
# =========================

def plot_visit_gmm(df_visit: pd.DataFrame, cluster_df: pd.DataFrame, outpath: Path) -> None:
    """
    Single phasor figure per visit:
    - phasor semicircle from PhasorPy
    - ROI means as scatter
    - cluster centers on same axes
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    plotted_any = False

    for bio_label in ["elastin", "cells", "melanin"]:
        d = df_visit[df_visit["bio_label"] == bio_label]
        if d.empty:
            continue

        # draw this group's scatter ON THE SAME AXES
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
            title=f"ROI phasor GMM - {df_visit['visit'].iloc[0]}",
            show=False,
        )
        plotted_any = True

    # if no groups got plotted, still create the phasor axes once
    if not plotted_any:
        plot_phasor(
            np.array([0.5]),
            np.array([0.0]),
            style="plot",
            marker="",
            linestyle="",
            frequency=PHASOR_FREQUENCY,
            ax=ax,
            title=f"ROI phasor GMM - {df_visit['visit'].iloc[0]}",
            show=False,
        )

    # cluster centers on same axes
    for _, row in cluster_df.iterrows():
        ax.scatter(
            row["g_mean"],
            row["s_mean"],
            s=140,
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


# =========================
# MAIN
# =========================

def main() -> None:
    print(f"[INFO] Buscando casos bajo: {ROOT_DIR}")
    cases = collect_cases(ROOT_DIR, PATIENT_FILTER)

    if not cases:
        raise RuntimeError("No encontré ningún phasor.tif con máscara asociada.")

    print(f"[INFO] Casos encontrados: {len(cases)}")
    for c in cases:
        print(f"  - {c.patient} | {c.visit} | {c.phasor_path}")

    all_roi_tables = []
    for case in cases:
        try:
            df_case = roi_table_from_case(case)
            if len(df_case) == 0:
                print(f"[WARN] Sin ROIs válidas en {case.phasor_path}")
                continue

            all_roi_tables.append(df_case)
            print(f"[OK] {case.visit} | {case.phasor_path.name}: {len(df_case)} ROIs")

        except Exception as e:
            print(f"[ERROR] Falló {case.phasor_path}: {e}")

    if not all_roi_tables:
        raise RuntimeError("No pude extraer ROIs válidas de ningún caso.")

    df_all_rois = pd.concat(all_roi_tables, ignore_index=True)

    roi_points_csv = OUTPUT_DIR / "roi_phasor_points_all.csv"
    df_all_rois.to_csv(roi_points_csv, index=False)
    print(f"[INFO] Guardado: {roi_points_csv}")

    all_cluster_stats = []
    all_roi_labeled = []

    for visit in sorted(df_all_rois["visit"].unique()):
        df_visit = df_all_rois[df_all_rois["visit"] == visit].copy()

        if len(df_visit) < 2:
            print(f"[WARN] Muy pocos puntos en {visit}, se omite GMM.")
            continue

        n_components = 2 if visit.lower() == "visit04" else 3
        print(f"[INFO] {visit}: GMM con {n_components} clusters sobre {len(df_visit)} ROIs")

        X = df_visit[["g_mean", "s_mean"]].to_numpy()

        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=0,
            n_init=10,
        )
        gmm.fit(X)

        df_visit_labeled, cluster_df = assign_biological_labels_by_phase(df_visit, gmm, visit)

        all_roi_labeled.append(df_visit_labeled)
        all_cluster_stats.append(cluster_df)

        plot_path = OUTPUT_DIR / f"{visit}_roi_phasor_gmm.png"
        plot_visit_gmm(df_visit_labeled, cluster_df, plot_path)
        print(f"[INFO] Plot guardado: {plot_path}")

    if all_cluster_stats:
        df_cluster_stats = pd.concat(all_cluster_stats, ignore_index=True)
        cluster_csv = OUTPUT_DIR / "phasor_cluster_stats_by_visit.csv"
        df_cluster_stats.to_csv(cluster_csv, index=False)
        print(f"[INFO] Guardado: {cluster_csv}")

    if all_roi_labeled:
        df_roi_labeled = pd.concat(all_roi_labeled, ignore_index=True)
        roi_labeled_csv = OUTPUT_DIR / "roi_phasor_points_with_gmm_labels.csv"
        df_roi_labeled.to_csv(roi_labeled_csv, index=False)
        print(f"[INFO] Guardado: {roi_labeled_csv}")

    print("[DONE]")


if __name__ == "__main__":
    main()