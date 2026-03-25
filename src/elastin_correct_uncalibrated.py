#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import imageio.v3 as iio
import numpy as np
import pandas as pd
import tifffile

from skimage.measure import label, regionprops
from sklearn.mixture import GaussianMixture


# =========================
# CONFIG
# =========================

ROOT_DIR = Path("/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449")
OUTPUT_DIR = ROOT_DIR / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATIENT_FILTER = None  # e.g. "p449" or None

# phasor stacks:
# 0 intensity
# 1 G first harmonic
# 2 S first harmonic
INTENSITY_IDX = 0
PHASOR_G_IDX = 1
PHASOR_S_IDX = 2

MIN_ROI_AREA = 1
FILTER_PHASOR_RANGE = False
G_MIN, G_MAX = -0.2, 1.2
S_MIN, S_MAX = -0.2, 1.2

VALID_INTENSITY_THRESHOLD = 0.0
OVERWRITE = True

# outputs
RAW_ROI_CSV = OUTPUT_DIR / "roi_phasor_points_uncalibrated.csv"
RAW_ROI_GMM_CSV = OUTPUT_DIR / "roi_phasor_points_with_gmm_labels_uncalibrated.csv"
RAW_CLUSTER_CSV = OUTPUT_DIR / "phasor_cluster_stats_by_visit_uncalibrated.csv"

COU_ROI_CSV = OUTPUT_DIR / "roi_phasor_points_coumarin.csv"
COU_ROI_GMM_CSV = OUTPUT_DIR / "roi_phasor_points_with_gmm_labels_coumarin.csv"
COU_CLUSTER_CSV = OUTPUT_DIR / "phasor_cluster_stats_by_visit_coumarin.csv"

ELASTIN_PARAMS_CSV = OUTPUT_DIR / "elastin_reference_params_raw_to_coumarin.csv"

OUTPUT_TIFF_NAME = "phasor_uncalibrated_elastin_corr.tif"


# =========================
# DATA STRUCTURES
# =========================

@dataclass
class Case:
    raw_phasor_path: Path
    coumarin_phasor_path: Path
    mask_path: Path
    patient: str
    visit: str


# =========================
# BASIC MATH
# =========================

def phasor_to_polar(g: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mod = np.sqrt(g**2 + s**2)
    phi = np.arctan2(s, g)
    return mod, phi


def polar_to_phasor(mod: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    g = mod * np.cos(phi)
    s = mod * np.sin(phi)
    return g, s


def circular_mean(angles_rad: np.ndarray) -> float:
    angles_rad = np.asarray(angles_rad, dtype=np.float64)
    return float(np.arctan2(np.mean(np.sin(angles_rad)),
                            np.mean(np.cos(angles_rad))))


def compute_phase(g: np.ndarray, s: np.ndarray) -> np.ndarray:
    return np.arctan2(s, g)


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


def find_matching_mask(mosaic_dir: Path) -> Optional[Path]:
    """
    Look for instance_mask_filtered in the mosaic folder and inside _new/.
    """
    direct_candidates = [
        mosaic_dir / "instance_mask_filtered.tif",
        mosaic_dir / "instance_mask_filtered.tiff",
        mosaic_dir / "instance_mask_filtered.png",
        mosaic_dir / "_new" / "instance_mask_filtered.tif",
        mosaic_dir / "_new" / "instance_mask_filtered.tiff",
        mosaic_dir / "_new" / "instance_mask_filtered.png",
    ]

    for c in direct_candidates:
        if c.exists():
            return c

    recursive = []
    recursive.extend(mosaic_dir.rglob("instance_mask_filtered.tif"))
    recursive.extend(mosaic_dir.rglob("instance_mask_filtered.tiff"))
    recursive.extend(mosaic_dir.rglob("instance_mask_filtered.png"))

    recursive = sorted(set(recursive))
    if recursive:
        return recursive[0]

    return None


def collect_cases(root_dir: Path, patient_filter: Optional[str] = None) -> List[Case]:
    """
    For each mosaic:
      - raw phasor:       phasor_uncalibrated.tif
      - coumarin phasor:  _new/phasor.tif
      - mask:             _new/instance_mask_filtered.*
    """
    cases: List[Case] = []

    for raw_phasor_path in root_dir.rglob("phasor_uncalibrated.tif"):
        if patient_filter is not None and patient_filter.lower() not in str(raw_phasor_path).lower():
            continue

        mosaic_dir = raw_phasor_path.parent
        coumarin_phasor_path = mosaic_dir / "_new" / "phasor.tif"
        mask_path = find_matching_mask(mosaic_dir)

        if not coumarin_phasor_path.exists():
            print(f"[WARN] Missing coumarin-calibrated phasor for: {raw_phasor_path}")
            continue

        if mask_path is None:
            print(f"[WARN] No encontré máscara para: {raw_phasor_path}")
            continue

        patient, visit = infer_patient_visit_from_path(raw_phasor_path)
        cases.append(
            Case(
                raw_phasor_path=raw_phasor_path,
                coumarin_phasor_path=coumarin_phasor_path,
                mask_path=mask_path,
                patient=patient,
                visit=visit,
            )
        )

    return sorted(cases, key=lambda c: (c.patient, c.visit, str(c.raw_phasor_path)))


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


def read_phasor_stack(phasor_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads phasor stack: (intensity, g, s)
    Expected format: (3, Y, X) or larger, using planes 0,1,2.
    """
    arr = tifffile.imread(phasor_path)
    arr = np.asarray(arr).squeeze()

    if arr.ndim != 3 or arr.shape[0] < 3:
        raise ValueError(
            f"No pude interpretar phasor TIFF con shape {arr.shape} en {phasor_path}"
        )

    intensity = arr[INTENSITY_IDX].astype(np.float64)
    g = arr[PHASOR_G_IDX].astype(np.float64)
    s = arr[PHASOR_S_IDX].astype(np.float64)
    return intensity, g, s


# =========================
# ROI -> ONE POINT
# =========================

def roi_table_from_stack(
    phasor_path: Path,
    mask_path: Path,
    patient: str,
    visit: str,
    phasor_kind: str,
) -> pd.DataFrame:
    """
    One ROI -> one phasor point (g_mean, s_mean) from a phasor stack.
    """
    intensity, g, s = read_phasor_stack(phasor_path)
    labels = read_mask(mask_path)

    if g.shape != labels.shape or s.shape != labels.shape:
        raise ValueError(
            f"Shape mismatch en {phasor_path.name} vs {mask_path.name}: "
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
        ivals = intensity[rr, cc]

        valid = np.isfinite(gvals) & np.isfinite(svals) & np.isfinite(ivals)

        if FILTER_PHASOR_RANGE:
            valid &= (
                (gvals >= G_MIN) & (gvals <= G_MAX) &
                (svals >= S_MIN) & (svals <= S_MAX)
            )

        if valid.sum() == 0:
            continue

        gvals = gvals[valid]
        svals = svals[valid]
        ivals = ivals[valid]

        rows.append({
            "patient": case_patient(patient),
            "visit": case_visit(visit),
            "phasor_kind": phasor_kind,
            "phasor_path": str(phasor_path),
            "mask_path": str(mask_path),
            "roi_label": int(roi_label),
            "area_px": int(area),
            "centroid_row": float(prop.centroid[0]),
            "centroid_col": float(prop.centroid[1]),
            "g_mean": float(np.mean(gvals)),
            "s_mean": float(np.mean(svals)),
            "intensity_mean": float(np.mean(ivals)),
            "g_std_within_roi": float(np.std(gvals, ddof=0)),
            "s_std_within_roi": float(np.std(svals, ddof=0)),
            "n_valid_pixels": int(valid.sum()),
        })

    return pd.DataFrame(rows)


def case_patient(patient: str) -> str:
    return patient


def case_visit(visit: str) -> str:
    return visit


# =========================
# CLUSTER BIOLOGY
# =========================

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


def n_components_for_visit(visit: str) -> int:
    return 2 if visit.lower() == "visit04" else 3


# =========================
# GMM PIPELINE
# =========================

def run_gmm_by_visit(
    df_all_rois: pd.DataFrame,
    output_roi_csv: Path,
    output_cluster_csv: Path,
    kind_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run GMM separately per visit for a given ROI table.
    """
    all_cluster_stats = []
    all_roi_labeled = []

    for visit in sorted(df_all_rois["visit"].unique()):
        df_visit = df_all_rois[df_all_rois["visit"] == visit].copy()

        if len(df_visit) < 2:
            print(f"[WARN] Muy pocos puntos en {visit}, se omite GMM para {kind_name}.")
            continue

        n_components = n_components_for_visit(visit)
        print(f"[INFO] {kind_name} | {visit}: GMM con {n_components} clusters sobre {len(df_visit)} ROIs")

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

    if not all_roi_labeled:
        raise RuntimeError(f"No pude generar labels GMM para {kind_name}.")

    df_roi_labeled = pd.concat(all_roi_labeled, ignore_index=True)
    df_roi_labeled.to_csv(output_roi_csv, index=False)

    df_cluster_stats = pd.concat(all_cluster_stats, ignore_index=True)
    df_cluster_stats.to_csv(output_cluster_csv, index=False)

    print(f"[INFO] Guardado ROI labeled ({kind_name}): {output_roi_csv}")
    print(f"[INFO] Guardado cluster stats ({kind_name}): {output_cluster_csv}")

    return df_roi_labeled, df_cluster_stats


# =========================
# ELASTIN REFERENCE
# =========================

def compute_elastin_reference_params(
    df_raw_labeled: pd.DataFrame,
    df_cou_labeled: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build correction parameters such that raw elastin is moved to the
    patient-level reference computed from coumarin-calibrated elastin.

    For each visit:
      measured = raw elastin centroid
      target   = mean(coumarin elastin centroids across visits)
    """
    df_raw_el = df_raw_labeled[df_raw_labeled["bio_label"] == "elastin"].copy()
    df_cou_el = df_cou_labeled[df_cou_labeled["bio_label"] == "elastin"].copy()

    if df_raw_el.empty:
        raise RuntimeError("No ROIs labeled as elastin in raw ROI table.")
    if df_cou_el.empty:
        raise RuntimeError("No ROIs labeled as elastin in coumarin ROI table.")

    raw_stats = (
        df_raw_el.groupby("visit")
        .agg(
            raw_centroid_g=("g_mean", "mean"),
            raw_centroid_s=("s_mean", "mean"),
            raw_sd_g=("g_mean", "std"),
            raw_sd_s=("s_mean", "std"),
            n_raw_elastin_rois=("g_mean", "size"),
        )
        .reset_index()
    )

    cou_stats = (
        df_cou_el.groupby("visit")
        .agg(
            cou_centroid_g=("g_mean", "mean"),
            cou_centroid_s=("s_mean", "mean"),
            cou_sd_g=("g_mean", "std"),
            cou_sd_s=("s_mean", "std"),
            n_cou_elastin_rois=("g_mean", "size"),
        )
        .reset_index()
    )

    raw_stats["raw_sd_g"] = raw_stats["raw_sd_g"].fillna(0.0)
    raw_stats["raw_sd_s"] = raw_stats["raw_sd_s"].fillna(0.0)
    cou_stats["cou_sd_g"] = cou_stats["cou_sd_g"].fillna(0.0)
    cou_stats["cou_sd_s"] = cou_stats["cou_sd_s"].fillna(0.0)

    # reference target comes from COUMARIN elastin centroids
    cou_mod_visit, cou_phi_visit = phasor_to_polar(
        cou_stats["cou_centroid_g"].to_numpy(),
        cou_stats["cou_centroid_s"].to_numpy(),
    )
    cou_stats["cou_mod_visit"] = cou_mod_visit
    cou_stats["cou_phi_visit"] = cou_phi_visit

    cou_mod_ref = float(np.mean(cou_mod_visit))
    cou_phi_ref = circular_mean(cou_phi_visit)

    # measured source comes from RAW elastin centroids
    raw_mod_visit, raw_phi_visit = phasor_to_polar(
        raw_stats["raw_centroid_g"].to_numpy(),
        raw_stats["raw_centroid_s"].to_numpy(),
    )
    raw_stats["raw_mod_visit"] = raw_mod_visit
    raw_stats["raw_phi_visit"] = raw_phi_visit

    # merge on visit
    out = raw_stats.merge(cou_stats, on="visit", how="inner")

    if out.empty:
        raise RuntimeError("No overlapping visits between raw elastin and coumarin elastin.")

    out["target_mod_ref"] = cou_mod_ref
    out["target_phi_ref"] = cou_phi_ref

    # correction that maps RAW visit elastin -> COUMARIN reference
    out["dphi"] = out["target_phi_ref"] - out["raw_phi_visit"]
    out["mod_scale"] = out["target_mod_ref"] / out["raw_mod_visit"]

    return out


# =========================
# APPLY CORRECTION TO TIFF
# =========================

def apply_elastin_correction_to_stack(
    phasor_path: Path,
    dphi: float,
    mod_scale: float,
    output_path: Path,
    valid_intensity_threshold: float = 0.0,
) -> None:
    """
    Apply correction to phasor_uncalibrated.tif and save:
      phasor_uncalibrated_elastin_corr.tif

    Input stack: (intensity, g, s)
    Output stack: (intensity, g_corr, s_corr)
    """
    arr = tifffile.imread(phasor_path)
    arr = np.asarray(arr)

    if arr.ndim != 3 or arr.shape[0] < 3:
        raise ValueError(f"Unexpected phasor stack shape {arr.shape} in {phasor_path}")

    intensity = arr[INTENSITY_IDX].astype(np.float64)
    g = arr[PHASOR_G_IDX].astype(np.float64)
    s = arr[PHASOR_S_IDX].astype(np.float64)

    valid = np.isfinite(intensity) & np.isfinite(g) & np.isfinite(s)
    valid &= intensity > valid_intensity_threshold

    mod, phi = phasor_to_polar(g, s)

    mod_corr = mod.copy()
    phi_corr = phi.copy()

    mod_corr[valid] = mod[valid] * mod_scale
    phi_corr[valid] = phi[valid] + dphi

    g_corr, s_corr = polar_to_phasor(mod_corr, phi_corr)

    out = np.stack(
        [
            intensity.astype(np.float32),
            g_corr.astype(np.float32),
            s_corr.astype(np.float32),
        ],
        axis=0,
    )

    tifffile.imwrite(output_path, out)


# =========================
# MAIN
# =========================

def main() -> None:
    print(f"[INFO] Buscando casos bajo: {ROOT_DIR}")
    cases = collect_cases(ROOT_DIR, PATIENT_FILTER)

    if not cases:
        raise RuntimeError("No encontré casos con raw phasor + coumarin phasor + máscara.")

    print(f"[INFO] Casos encontrados: {len(cases)}")
    for c in cases:
        print(f"  - {c.patient} | {c.visit}")
        print(f"    raw  -> {c.raw_phasor_path}")
        print(f"    cou  -> {c.coumarin_phasor_path}")
        print(f"    mask -> {c.mask_path}")

    # ---------------------------------
    # 1) ROI tables from RAW and COUMARIN
    # ---------------------------------
    all_raw_roi_tables = []
    all_cou_roi_tables = []

    for case in cases:
        try:
            df_raw = roi_table_from_stack(
                case.raw_phasor_path,
                case.mask_path,
                patient=case.patient,
                visit=case.visit,
                phasor_kind="raw",
            )
            df_cou = roi_table_from_stack(
                case.coumarin_phasor_path,
                case.mask_path,
                patient=case.patient,
                visit=case.visit,
                phasor_kind="coumarin",
            )

            if len(df_raw) == 0:
                print(f"[WARN] Sin ROIs válidas en RAW {case.raw_phasor_path}")
            else:
                all_raw_roi_tables.append(df_raw)

            if len(df_cou) == 0:
                print(f"[WARN] Sin ROIs válidas en COUMARIN {case.coumarin_phasor_path}")
            else:
                all_cou_roi_tables.append(df_cou)

            print(f"[OK] {case.visit} | {case.raw_phasor_path.parent.name}")

        except Exception as e:
            print(f"[ERROR] Falló {case.raw_phasor_path}: {e}")

    if not all_raw_roi_tables:
        raise RuntimeError("No pude extraer ROIs válidas de ningún RAW.")
    if not all_cou_roi_tables:
        raise RuntimeError("No pude extraer ROIs válidas de ningún COUMARIN.")

    df_raw_all = pd.concat(all_raw_roi_tables, ignore_index=True)
    df_cou_all = pd.concat(all_cou_roi_tables, ignore_index=True)

    df_raw_all.to_csv(RAW_ROI_CSV, index=False)
    df_cou_all.to_csv(COU_ROI_CSV, index=False)
    print(f"[INFO] Guardado: {RAW_ROI_CSV}")
    print(f"[INFO] Guardado: {COU_ROI_CSV}")

    # ---------------------------------
    # 2) GMM separately for RAW and COUMARIN
    # ---------------------------------
    df_raw_labeled, df_raw_clusters = run_gmm_by_visit(
        df_raw_all,
        output_roi_csv=RAW_ROI_GMM_CSV,
        output_cluster_csv=RAW_CLUSTER_CSV,
        kind_name="raw",
    )

    df_cou_labeled, df_cou_clusters = run_gmm_by_visit(
        df_cou_all,
        output_roi_csv=COU_ROI_GMM_CSV,
        output_cluster_csv=COU_CLUSTER_CSV,
        kind_name="coumarin",
    )

    # ---------------------------------
    # 3) Build correction from RAW elastin -> COUMARIN elastin reference
    # ---------------------------------
    correction_table = compute_elastin_reference_params(df_raw_labeled, df_cou_labeled)
    correction_table.to_csv(ELASTIN_PARAMS_CSV, index=False)
    print(f"[INFO] Guardado: {ELASTIN_PARAMS_CSV}")
    print("\n=== RAW -> COUMARIN elastin correction params ===")
    print(correction_table[[
        "visit",
        "raw_centroid_g", "raw_centroid_s",
        "cou_centroid_g", "cou_centroid_s",
        "target_mod_ref", "target_phi_ref",
        "dphi", "mod_scale"
    ]])

    correction_map = {
        row["visit"]: (float(row["dphi"]), float(row["mod_scale"]))
        for _, row in correction_table.iterrows()
    }

    # ---------------------------------
    # 4) Apply correction to RAW TIFFs
    # ---------------------------------
    ok = 0
    fail = 0

    for case in cases:
        if case.visit not in correction_map:
            print(f"[WARN] No correction params for {case.visit}, skipping {case.raw_phasor_path}")
            continue

        dphi, mod_scale = correction_map[case.visit]
        out_path = case.raw_phasor_path.with_name(OUTPUT_TIFF_NAME)

        if out_path.exists() and not OVERWRITE:
            print(f"[SKIP] Exists: {out_path}")
            continue

        try:
            apply_elastin_correction_to_stack(
                phasor_path=case.raw_phasor_path,
                dphi=dphi,
                mod_scale=mod_scale,
                output_path=out_path,
                valid_intensity_threshold=VALID_INTENSITY_THRESHOLD,
            )
            print(
                f"[OK] {case.visit} -> {out_path.name} "
                f"(dphi={dphi:.6f}, mod_scale={mod_scale:.6f})"
            )
            ok += 1
        except Exception as e:
            print(f"[ERROR] No pude corregir {case.raw_phasor_path}")
            print(f"        {type(e).__name__}: {e}")
            fail += 1

    print(f"\n[DONE] corrected ok={ok} fail={fail}")


if __name__ == "__main__":
    main()