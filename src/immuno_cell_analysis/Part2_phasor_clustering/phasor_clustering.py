"""
Part 2 — Phasor clustering (GMM) using per-instance means from Part 1 CSVs.

Inputs (produced by Part 1, inside each mask_* folder):
  - mask_instances_features_minEqDiam8px_tau0-12.csv

This script:
  - Finds all those CSVs recursively under PATIENT_DIR
  - Runs GMM clustering in phasor space using columns:
        g_mean, s_mean
  - Auto-labels clusters:
        3 clusters: by phasor angle of cluster COM:
            smallest angle -> melanin
            middle angle   -> cell
            largest angle  -> elastin
        2 clusters: melanin absent; use tau proxy:
            lower tau -> cell
            higher tau -> elastin
  - Saves, per CSV:
        * clustered CSV with phasor_class_name
        * ONLY_cells CSV
        * phasor segmented JPG plot (PhasorPlot)
  - Saves a global summary CSV.

Notes:
  - This works directly on the final Part 1 CSVs (tau-filtered).
  - No raw image needed here.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from phasorpy.plot import PhasorPlot


# ============================================================
# CONFIG — EDIT THESE
# ============================================================
PATIENT_DIR = Path("/Users/schutyb/Documents/balu_lab/data_patient_449")

# Where to write Part 2 outputs
OUT_ROOT = PATIENT_DIR / "part2_phasor_clustering_out"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Input CSV filename (from Part 1)
CSV_NAME = "mask_instances_features_minEqDiam8px_tau0-12.csv"

# Columns in your Part 1 CSV (confirmed from your code)
COL_G = "g_mean"
COL_S = "s_mean"

# (Optional) if you want to use tau_phase_mean_ns for diagnostics or 2-cluster labeling,
# we compute tau proxy from phasor (G,S) by default (same as your old code).
FREQUENCY_MHZ = 80.0

# GMM settings
N_CLASSES_DEFAULT = 3
N_CLASSES_SPECIAL = 2
RANDOM_STATE = 0
COV_TYPE = "full"

# If you have specific mosaics (or visits) that must use 2 clusters:
# Put either mosaic folder names or any substring you want to match in the CSV path.
# Example: {"visit_04/Mosaic07_4x4_FOV600_z150_32Sp"} or {"Mosaic07_"}
TWO_CLUSTER_MATCH = {
    # "visit_04/Mosaic07_4x4_FOV600_z150_32Sp",
}

# Plotting
CLASS_COLORS = {
    "melanin": "#1f77b4",
    "cell": "#ff7f0e",
    "elastin": "#2ca02c",
}
MARKER_SIZE = 6
ALPHA = 0.45
EXPORT_DPI = 300
JPG_QUALITY = 95
JPG_SUBSAMPLING = 0


# ============================================================
# HELPERS
# ============================================================
def safe_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def tau_proxy_from_phasor(G: np.ndarray, S: np.ndarray, freq_mhz: float) -> np.ndarray:
    """
    Lifetime proxy from phasor coordinates (single-exponential monotonic proxy).
    tau ≈ (S/G)/omega
    """
    omega = 2.0 * np.pi * (freq_mhz * 1e6)
    eps = 1e-12
    return (S / (G + eps)) / omega


def choose_n_classes(csv_path: Path) -> int:
    s = str(csv_path)
    for key in TWO_CLUSTER_MATCH:
        if key in s:
            return N_CLASSES_SPECIAL
    return N_CLASSES_DEFAULT


def relabel_clusters(
    G: np.ndarray,
    S: np.ndarray,
    labels_raw: np.ndarray,
    n_classes: int,
    freq_mhz: float,
):
    """
    Returns:
      label_map: dict raw_label -> {"melanin","cell","elastin"} (or only cell/elastin for 2 clusters)
      centers_sorted: list tuples for printing
      sort_mode: "angle" or "tau_proxy"
    """
    unique_labels = np.unique(labels_raw)

    if n_classes == 2:
        centers = []
        for lab in unique_labels:
            sel = labels_raw == lab
            Gc = float(np.mean(G[sel]))
            Sc = float(np.mean(S[sel]))
            tau_c = float(tau_proxy_from_phasor(np.array([Gc]), np.array([Sc]), freq_mhz=freq_mhz)[0])
            centers.append((int(lab), tau_c, Gc, Sc))

        centers_sorted = sorted(centers, key=lambda t: t[1])  # low tau -> cell, high tau -> elastin
        ordered_names = ["cell", "elastin"]
        label_map = {lab: name for (lab, _, _, _), name in zip(centers_sorted, ordered_names)}
        return label_map, centers_sorted, "tau_proxy"

    # Default: 3 clusters by phasor angle
    centers = []
    for lab in unique_labels:
        sel = labels_raw == lab
        Gc = float(np.mean(G[sel]))
        Sc = float(np.mean(S[sel]))
        angle = float(np.arctan2(Sc, Gc))
        centers.append((int(lab), angle, Gc, Sc))

    centers_sorted = sorted(centers, key=lambda t: t[1])  # small -> melanin, mid -> cell, large -> elastin
    ordered_names = ["melanin", "cell", "elastin"]
    label_map = {lab: name for (lab, _, _, _), name in zip(centers_sorted, ordered_names)}
    return label_map, centers_sorted, "angle"


def save_phasorplot_high_quality(plot_obj, out_path: Path,
                                 dpi: int = 300,
                                 quality: int = 95,
                                 subsampling: int = 0):
    """
    Save PhasorPlot with good raster quality (JPG).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Try PhasorPy save (supports dpi/quality)
    try:
        plot_obj.save(str(out_path), dpi=dpi, quality=quality)
        return
    except Exception:
        pass

    # Fallback: matplotlib fig if available
    fig = getattr(plot_obj, "fig", None) or getattr(plot_obj, "figure", None)
    if fig is not None:
        fig.savefig(
            out_path,
            dpi=dpi,
            bbox_inches="tight",
            pil_kwargs={"quality": int(quality), "subsampling": int(subsampling)},
        )
        return

    # Last fallback
    plot_obj.save(str(out_path))


def infer_visit_mosaic_from_path(csv_path: Path):
    """
    Try to infer visit + mosaic names from a path like:
      .../visit_01/MosaicXX.../mask_.../mask_instances_features_...csv
    """
    parts = csv_path.parts
    visit = None
    mosaic = None
    for p in parts:
        if p.startswith("visit_"):
            visit = p
        if p.startswith("Mosaic"):
            mosaic = p
    return visit, mosaic


def relative_output_dir(csv_path: Path) -> Path:
    """
    Mirror input structure under OUT_ROOT, using visit/mosaic if found.
    Otherwise, uses parent folder name.
    """
    visit, mosaic = infer_visit_mosaic_from_path(csv_path)
    if visit and mosaic:
        return OUT_ROOT / visit / mosaic
    # fallback: just use the directory name containing the csv
    return OUT_ROOT / csv_path.parent.name


# ============================================================
# MAIN
# ============================================================
def main():
    csv_paths = sorted(PATIENT_DIR.rglob(CSV_NAME))
    if not csv_paths:
        raise FileNotFoundError(f"No '{CSV_NAME}' found under: {PATIENT_DIR}")

    print(f"[INFO] Patient root: {PATIENT_DIR}")
    print(f"[INFO] Found {len(csv_paths)} CSV(s) matching {CSV_NAME}")
    print(f"[INFO] Output root: {OUT_ROOT}")

    summary_rows = []

    for csv_path in csv_paths:
        out_dir = relative_output_dir(csv_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        visit, mosaic = infer_visit_mosaic_from_path(csv_path)
        tag = f"{visit}/{mosaic}" if (visit and mosaic) else csv_path.parent.name

        n_classes = choose_n_classes(csv_path)

        print(f"\n=== {tag} | clusters={n_classes} ===")
        print(f"CSV: {csv_path}")

        df = pd.read_csv(csv_path)

        if COL_G not in df.columns or COL_S not in df.columns:
            print(f"  [WARN] Missing columns {COL_G} or {COL_S}. Skipping.")
            continue

        G = safe_numeric(df[COL_G])
        S = safe_numeric(df[COL_S])
        valid = np.isfinite(G) & np.isfinite(S)

        Gv, Sv = G[valid], S[valid]
        X = np.column_stack([Gv, Sv])

        if X.shape[0] < 50:
            print(f"  [WARN] Too few valid phasor points (N={X.shape[0]}). Skipping.")
            continue

        # --- GMM ---
        gmm = GaussianMixture(
            n_components=n_classes,
            covariance_type=COV_TYPE,
            random_state=RANDOM_STATE,
        )
        labels_raw = gmm.fit_predict(X)

        # --- Auto-label ---
        label_map, centers_sorted, sort_mode = relabel_clusters(
            Gv, Sv, labels_raw, n_classes=n_classes, freq_mhz=FREQUENCY_MHZ
        )
        labels_name = np.array([label_map[int(l)] for l in labels_raw], dtype=object)

        counts = {
            "melanin": int(np.sum(labels_name == "melanin")),
            "cell": int(np.sum(labels_name == "cell")),
            "elastin": int(np.sum(labels_name == "elastin")),
        }

        print(f"Cluster centers sorted by {sort_mode}:")
        if sort_mode == "tau_proxy":
            for lab, tau_c, gc, sc in centers_sorted:
                print(f"  raw {lab} | tau_proxy={tau_c:.3e} s | (G,S)=({gc:.4f},{sc:.4f}) -> {label_map[lab]}")
        else:
            for lab, ang, gc, sc in centers_sorted:
                print(f"  raw {lab} | angle={ang:.4f} | (G,S)=({gc:.4f},{sc:.4f}) -> {label_map[lab]}")
        print("Counts:", counts)

        # --- Write labels back into full df ---
        df["phasor_class_name"] = None
        df.loc[valid, "phasor_class_name"] = labels_name

        # --- Save outputs ---
        stem = f"{visit}_{mosaic}" if (visit and mosaic) else csv_path.parent.name

        out_clustered = out_dir / f"{stem}_clustered.csv"
        out_cells = out_dir / f"{stem}_ONLY_cells.csv"
        out_plot = out_dir / f"{stem}_phasor_segmented.jpg"

        df.to_csv(out_clustered, index=False)

        df_cells = df[df["phasor_class_name"] == "cell"].copy()
        df_cells.to_csv(out_cells, index=False)

        # --- Plot ---
        plot = PhasorPlot(frequency=FREQUENCY_MHZ, title=stem)

        for cls in ["melanin", "cell", "elastin"]:
            sel = labels_name == cls
            if np.sum(sel) == 0:
                continue
            plot.plot(
                Gv[sel],
                Sv[sel],
                marker=".",
                markersize=MARKER_SIZE,
                alpha=ALPHA,
                label=cls,
                color=CLASS_COLORS[cls],
            )
        plot.legend()

        save_phasorplot_high_quality(
            plot,
            out_plot,
            dpi=EXPORT_DPI,
            quality=JPG_QUALITY,
            subsampling=JPG_SUBSAMPLING,
        )

        print(f"  [SAVED] clustered: {out_clustered}")
        print(f"  [SAVED] cells-only: {out_cells} (N={len(df_cells)})")
        print(f"  [SAVED] plot: {out_plot}")

        summary_rows.append({
            "visit": visit,
            "mosaic": mosaic,
            "tag": tag,
            "input_csv": str(csv_path),
            "n_total_rows": int(len(df)),
            "n_valid_phasor": int(X.shape[0]),
            "n_classes_used": int(n_classes),
            "n_melanin": counts["melanin"],
            "n_cell": counts["cell"],
            "n_elastin": counts["elastin"],
            "clustered_csv": str(out_clustered),
            "cells_only_csv": str(out_cells),
            "phasor_plot_jpg": str(out_plot),
            "raw_label_to_name": str(label_map),
            "sort_mode": sort_mode,
        })

    # --- Global summary ---
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = OUT_ROOT / "summary_counts.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(f"\n✅ Done. Summary saved: {summary_csv}")
    print(f"📁 Outputs in: {OUT_ROOT}")


if __name__ == "__main__":
    main()