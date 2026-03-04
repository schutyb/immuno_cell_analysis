#!/usr/bin/env python3
"""
UMAP + clustering of CELL structures from per-mosaic CSVs.

- Recursively finds all structure_features_phasor_classified.csv under ROOT/visit_*/Mosaic*/
- Filters rows where class == "cell"
- Builds feature matrix (GREEN primary; optional BLUE)
- Scales features (RobustScaler)
- Runs UMAP to 2D for visualization
- Runs clustering:
    Option 1: GMM with BIC-based K selection
    Option 2: HDBSCAN (optional; if installed)

Outputs (saved next to each input CSV):
- <stem>_CELL_umap_clusters.csv          (adds: cell_subcluster, umap1, umap2)
- <stem>_CELL_umap.png                  (UMAP scatter colored by cluster)
- <stem>_CELL_phasor_by_cluster.png     (phasor G/S colored by cluster)
- <stem>_CELL_cluster_summary.csv       (per-cluster means/counts)

Also saves a global summary:
ROOT/umap_cell_out/summary_all_mosaics.csv

EDIT ROOT below and run:
  python umap_cluster_cells_batch.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.mixture import GaussianMixture

# UMAP is an extra dependency: pip install umap-learn
import umap

# optional HDBSCAN: pip install hdbscan
try:
    import hdbscan  # type: ignore
    HAS_HDBSCAN = True
except Exception:
    HAS_HDBSCAN = False


# ============================================================
# CONFIG (EDIT ME)
# ============================================================
ROOT = Path("/Users/schutyb/Documents/balu_lab/immuno_cell_analysis_data")

CSV_NAME = "structure_features_phasor_classified.csv"

# class labels
CLASS_COL_CANDIDATES = ["phasor_class_name", "phasor_class"]
CELL_LABEL = "cell"

# Features (GREEN primary)
GREEN_FEATURES = [
    "green_g_mean",
    "green_s_mean",
    "green_tau_phase_mean_ns",
    "green_tau_mod_mean_ns",
    "green_modulation_mean",
]

# Optional BLUE features (set USE_BLUE=False to ignore)
USE_BLUE = False
BLUE_FEATURES = [
    "blue_g_mean",
    "blue_s_mean",
    "blue_tau_phase_mean_ns",
    "blue_tau_mod_mean_ns",
    "blue_modulation_mean",
]

# Morphology features (optional)
USE_MORPHOLOGY = True
MORPH_FEATURES = [
    "area_px",
    "equivalent_diameter_px",
    "perimeter_px",
    "major_axis_length_px",
    "minor_axis_length_px",
    "axis_ratio",        # if you have it
    "circularity",       # if you have it
]

# Intensity filter (optional)
USE_INTENSITY_FILTER = True
GREEN_INTENSITY_COL = "green_intensity_mean"
MIN_GREEN_INTENSITY = 1.0

# UMAP
UMAP_N_NEIGHBORS = 20
UMAP_MIN_DIST = 0.10
UMAP_METRIC = "euclidean"
UMAP_RANDOM_STATE = 0

# Clustering method: "gmm" or "hdbscan"
CLUSTER_METHOD = "gmm"

# GMM settings
GMM_K_MIN = 2
GMM_K_MAX = 6
GMM_COV_TYPE = "full"
GMM_RANDOM_STATE = 0

# HDBSCAN settings (only if installed)
HDBSCAN_MIN_CLUSTER_SIZE = 30
HDBSCAN_MIN_SAMPLES = 10

# Output folder for global summary
GLOBAL_OUT = ROOT / "umap_cell_out"

# Plotting
MAX_POINTS_PLOT = 100_000   # subsample for plotting speed
PLOT_DPI = 300
# ============================================================


# -------------------------
# Helpers
# -------------------------
def find_class_col(df: pd.DataFrame) -> str:
    for c in CLASS_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"No class column found. Tried {CLASS_COL_CANDIDATES}. Found: {list(df.columns)}")


def discover_csvs(root: Path) -> list[Path]:
    return sorted(root.rglob(CSV_NAME))


def safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def add_log_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "area_px" in df.columns:
        df["log_area_px"] = np.log1p(df["area_px"].astype(float))
    if "perimeter_px" in df.columns:
        df["log_perimeter_px"] = np.log1p(df["perimeter_px"].astype(float))
    return df


def choose_feature_columns(df: pd.DataFrame) -> list[str]:
    feats = []
    # GREEN
    feats += [c for c in GREEN_FEATURES if c in df.columns]

    # BLUE optional
    if USE_BLUE:
        feats += [c for c in BLUE_FEATURES if c in df.columns]

    # Morphology optional (also add log_area/log_perimeter if possible)
    if USE_MORPHOLOGY:
        df2 = add_log_features(df)
        # add logs if created
        if "log_area_px" in df2.columns:
            feats.append("log_area_px")
        if "log_perimeter_px" in df2.columns:
            feats.append("log_perimeter_px")
        # add the rest if present
        feats += [c for c in MORPH_FEATURES if c in df2.columns]
        return feats, df2

    return feats, df


def subsample_for_plot(df: pd.DataFrame, max_points: int, seed: int = 0) -> pd.DataFrame:
    if max_points is None or len(df) <= max_points:
        return df
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=max_points, replace=False)
    return df.iloc[idx].copy()


def gmm_best_k(X: np.ndarray, kmin: int, kmax: int) -> tuple[int, np.ndarray]:
    best_k = None
    best_bic = np.inf
    best_labels = None

    for k in range(kmin, kmax + 1):
        gmm = GaussianMixture(
            n_components=int(k),
            covariance_type=GMM_COV_TYPE,
            random_state=GMM_RANDOM_STATE,
        )
        gmm.fit(X)
        bic = gmm.bic(X)
        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_labels = gmm.predict(X)

    assert best_k is not None and best_labels is not None
    return best_k, best_labels


def run_hdbscan(X: np.ndarray) -> np.ndarray:
    if not HAS_HDBSCAN:
        raise RuntimeError("hdbscan is not installed. Install with: pip install hdbscan")
    cl = hdbscan.HDBSCAN(
        min_cluster_size=int(HDBSCAN_MIN_CLUSTER_SIZE),
        min_samples=int(HDBSCAN_MIN_SAMPLES),
    )
    return cl.fit_predict(X)  # -1 = noise


def plot_umap(df_plot: pd.DataFrame, out_png: Path, title: str):
    fig = plt.figure(figsize=(7, 6), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    sc = ax.scatter(
        df_plot["umap1"], df_plot["umap2"],
        c=df_plot["cell_subcluster"].astype(int),
        s=6, alpha=0.75
    )
    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    fig.colorbar(sc, ax=ax, label="cluster")
    fig.savefig(out_png, dpi=PLOT_DPI)
    plt.close(fig)


def plot_phasor_gs(df_plot: pd.DataFrame, out_png: Path, title: str):
    # Prefer GREEN g/s
    if "green_g_mean" not in df_plot.columns or "green_s_mean" not in df_plot.columns:
        return

    fig = plt.figure(figsize=(6.5, 6), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    sc = ax.scatter(
        df_plot["green_g_mean"], df_plot["green_s_mean"],
        c=df_plot["cell_subcluster"].astype(int),
        s=8, alpha=0.65
    )
    ax.set_title(title)
    ax.set_xlabel("g (GREEN)")
    ax.set_ylabel("s (GREEN)")
    ax.set_aspect("equal", "box")
    fig.colorbar(sc, ax=ax, label="cluster")
    fig.savefig(out_png, dpi=PLOT_DPI)
    plt.close(fig)


def cluster_summary(df: pd.DataFrame, out_csv: Path, feature_cols: list[str]):
    grp = df.groupby("cell_subcluster", dropna=False)
    summary = grp[feature_cols].mean(numeric_only=True)
    summary["N"] = grp.size()
    summary = summary.sort_index()
    summary.to_csv(out_csv, index=True)


# -------------------------
# Main
# -------------------------
def main():
    GLOBAL_OUT.mkdir(parents=True, exist_ok=True)

    csvs = discover_csvs(ROOT)
    if not csvs:
        raise FileNotFoundError(f"No '{CSV_NAME}' found under: {ROOT}")

    print(f"[INFO] Found {len(csvs)} CSV(s)")

    global_rows = []

    for csv_path in csvs:
        df0 = pd.read_csv(csv_path)
        class_col = find_class_col(df0)

        df0[class_col] = df0[class_col].astype(str).str.strip().str.lower()
        df = df0[df0[class_col] == CELL_LABEL].copy()

        if df.empty:
            print(f"[WARN] {csv_path.name}: no 'cell' rows. Skipping.")
            continue

        # intensity filter
        if USE_INTENSITY_FILTER and GREEN_INTENSITY_COL in df.columns:
            df[GREEN_INTENSITY_COL] = pd.to_numeric(df[GREEN_INTENSITY_COL], errors="coerce")
            df = df[df[GREEN_INTENSITY_COL] > float(MIN_GREEN_INTENSITY)].copy()

        if df.empty:
            print(f"[WARN] {csv_path.name}: empty after intensity filter. Skipping.")
            continue

        feats, df2 = choose_feature_columns(df)
        if len(feats) < 2:
            print(f"[WARN] {csv_path.name}: not enough features found. Have: {feats}")
            continue

        df2 = safe_numeric(df2, feats)

        # drop rows with NaNs in features
        Xdf = df2[feats].copy()
        valid = np.isfinite(Xdf.to_numpy(dtype=float)).all(axis=1)
        df2 = df2.loc[valid].copy()
        X = df2[feats].to_numpy(dtype=float)

        if X.shape[0] < 50:
            print(f"[WARN] {csv_path.name}: too few valid cell rows (N={X.shape[0]}). Skipping.")
            continue

        # scale
        scaler = RobustScaler()
        Xs = scaler.fit_transform(X)

        # UMAP
        reducer = umap.UMAP(
            n_neighbors=int(UMAP_N_NEIGHBORS),
            min_dist=float(UMAP_MIN_DIST),
            metric=str(UMAP_METRIC),
            random_state=int(UMAP_RANDOM_STATE),
        )
        emb = reducer.fit_transform(Xs)
        df2["umap1"] = emb[:, 0]
        df2["umap2"] = emb[:, 1]

        # clustering
        if CLUSTER_METHOD.lower() == "hdbscan":
            labels = run_hdbscan(Xs)
            k_used = len(set(labels)) - (1 if -1 in labels else 0)
            method_desc = f"HDBSCAN(k~{k_used})"
        else:
            k_used, labels = gmm_best_k(Xs, GMM_K_MIN, GMM_K_MAX)
            method_desc = f"GMM(BIC,K={k_used})"

        df2["cell_subcluster"] = labels.astype(int)

        # outputs next to csv
        out_csv = csv_path.with_name(f"{csv_path.stem}_CELL_umap_clusters.csv")
        out_umap_png = csv_path.with_name(f"{csv_path.stem}_CELL_umap.png")
        out_phasor_png = csv_path.with_name(f"{csv_path.stem}_CELL_phasor_by_cluster.png")
        out_summary = csv_path.with_name(f"{csv_path.stem}_CELL_cluster_summary.csv")

        df2.to_csv(out_csv, index=False)

        # plots (subsample)
        dfp = subsample_for_plot(df2, MAX_POINTS_PLOT, seed=0)
        plot_umap(dfp, out_umap_png, title=f"{csv_path.parent.name} | cells | {method_desc}")
        plot_phasor_gs(dfp, out_phasor_png, title=f"{csv_path.parent.name} | GREEN phasor | {method_desc}")

        # summary
        cluster_summary(df2, out_summary, feature_cols=[c for c in feats if c in df2.columns])

        print(f"[OK] {csv_path.parent.name}: cells={len(df2):,} | {method_desc}")
        print(f"     saved: {out_csv.name}")
        print(f"     plots: {out_umap_png.name}, {out_phasor_png.name}")

        global_rows.append({
            "csv_path": str(csv_path),
            "visit": next((p for p in csv_path.parts if p.startswith("visit_")), "visit_unknown"),
            "mosaic": next((p for p in csv_path.parts if p.startswith("Mosaic")), csv_path.parent.name),
            "n_cells": int(len(df2)),
            "n_features": int(len(feats)),
            "features_used": ",".join(feats),
            "cluster_method": method_desc,
            "k_used": int(k_used) if isinstance(k_used, (int, np.integer)) else None,
            "out_csv": str(out_csv),
            "out_umap_png": str(out_umap_png),
            "out_phasor_png": str(out_phasor_png),
            "out_summary_csv": str(out_summary),
        })

    # global summary
    if global_rows:
        gdf = pd.DataFrame(global_rows)
        g_out = GLOBAL_OUT / "summary_all_mosaics.csv"
        gdf.to_csv(g_out, index=False)
        print(f"\n✅ Global summary saved:\n  {g_out}")
    else:
        print("\n[WARN] No outputs produced (no valid cell rows found).")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()