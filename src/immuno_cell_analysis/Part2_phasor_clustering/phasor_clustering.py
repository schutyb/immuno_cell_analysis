"""
Part 2 — Phasor clustering (GMM) on per-instance mean phasor coordinates.

Reads all per-mosaic CSVs produced by Part 1:
  mask_instances_features_minEqDiam8px_tau0-12.csv

Default behavior:
- For visit_01..visit_03: GMM with 3 clusters -> auto-label by phasor angle:
    smallest angle -> melanin
    middle angle   -> cell
    largest angle  -> elastin

Special case:
- For visit_04: GMM with 2 clusters (melanin absent) -> auto-label using tau_phase_mean_ns:
    lower tau -> cell
    higher tau -> elastin

Outputs:
- Per input CSV:
    * <stem>_phasor_classified.csv   (all objects + 'phasor_class_name')
    * <stem>_ONLY_cells.csv         (only rows classified as cell)
    * <stem>_phasor_segmented.jpg   (phasor segmented plot)
- Global:
    * summary_counts.csv

Notes:
- Assumes CSV columns:
    g_mean, s_mean, tau_phase_mean_ns (for visit_04 labeling), and label (instance id)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from phasorpy.plot import PhasorPlot


# ============================================================
# CONFIG
# ============================================================
PATIENT_DIR = Path("/Users/schutyb/Documents/balu_lab/data_patient_449")

# find these CSVs recursively under PATIENT_DIR
CSV_NAME = "mask_instances_features_minEqDiam8px_tau0-12.csv"

# where to put outputs (centralized)
OUT_ROOT = PATIENT_DIR / "phasor_cluster_out"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# phasor columns (from Part 1)
COL_G = "g_mean"
COL_S = "s_mean"

# tau column used ONLY for visit_04 2-cluster labeling
COL_TAU = "tau_phase_mean_ns"

# GMM
N_CLASSES_DEFAULT = 3
N_CLASSES_VISIT4 = 2
RANDOM_STATE = 0
COV_TYPE = "full"

# plotting
FREQUENCY_MHZ = 80.0
MARKER_SIZE = 6
ALPHA = 0.45
EXPORT_DPI = 300
JPG_QUALITY = 95
JPG_SUBSAMPLING = 0  # 0 = best quality

# fixed colors per class
CLASS_COLORS = {
    "melanin": "#1f77b4",
    "cell":    "#ff7f0e",
    "elastin": "#2ca02c",
}


# ============================================================
# Helpers
# ============================================================
def safe_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def discover_csvs(root: Path) -> list[Path]:
    return sorted(root.rglob(CSV_NAME))


def parse_visit_and_mosaic(csv_path: Path) -> tuple[str, str]:
    """
    Expects paths like:
      .../visit_XX/MosaicYY.../mask_.../mask_instances_features_minEqDiam8px_tau0-12.csv
    Returns: (visit_name, mosaic_name)
    """
    parts = list(csv_path.parts)
    visit = next((p for p in parts if p.startswith("visit_")), "visit_unknown")
    mosaic = next((p for p in parts if p.startswith("Mosaic")), "Mosaic_unknown")
    return visit, mosaic


def choose_n_classes(visit_name: str) -> int:
    # Force visit_04 -> 2 clusters
    if visit_name == "visit_04":
        return N_CLASSES_VISIT4
    return N_CLASSES_DEFAULT


def relabel_clusters_3_by_angle(G: np.ndarray, S: np.ndarray, labels_raw: np.ndarray):
    """
    Sort clusters by angle atan2(S,G):
      smallest angle -> melanin
      middle angle   -> cell
      largest angle  -> elastin
    """
    unique = np.unique(labels_raw)
    centers = []
    for lab in unique:
        sel = labels_raw == lab
        Gc = float(np.mean(G[sel]))
        Sc = float(np.mean(S[sel]))
        ang = float(np.arctan2(Sc, Gc))
        centers.append((int(lab), ang, Gc, Sc))
    centers_sorted = sorted(centers, key=lambda t: t[1])
    ordered_names = ["melanin", "cell", "elastin"]
    label_map = {lab: name for (lab, _, _, _), name in zip(centers_sorted, ordered_names)}
    return label_map, centers_sorted, "angle"


def relabel_clusters_2_by_tau(labels_raw: np.ndarray, tau: np.ndarray):
    """
    For 2 clusters (visit_04):
      lower mean tau -> cell
      higher mean tau -> elastin
    """
    unique = np.unique(labels_raw)
    centers = []
    for lab in unique:
        sel = labels_raw == lab
        tau_c = float(np.nanmean(tau[sel]))
        centers.append((int(lab), tau_c))
    centers_sorted = sorted(centers, key=lambda t: t[1])
    ordered_names = ["cell", "elastin"]
    label_map = {lab: name for (lab, _), name in zip(centers_sorted, ordered_names)}
    return label_map, centers_sorted, "tau_phase_mean"


def save_phasorplot_high_quality(plot_obj: PhasorPlot, out_path: Path):
    """
    Save PhasorPlot with high-quality JPG output.
    """
    try:
        plot_obj.save(str(out_path), dpi=EXPORT_DPI, quality=JPG_QUALITY)
        return
    except Exception:
        pass

    fig = getattr(plot_obj, "fig", None) or getattr(plot_obj, "figure", None)
    if fig is not None:
        fig.savefig(
            out_path,
            dpi=EXPORT_DPI,
            bbox_inches="tight",
            pil_kwargs={"quality": int(JPG_QUALITY), "subsampling": int(JPG_SUBSAMPLING)},
        )
        return

    plot_obj.save(str(out_path))


# ============================================================
# Main
# ============================================================
def main():
    csv_files = discover_csvs(PATIENT_DIR)
    if not csv_files:
        raise FileNotFoundError(f"No '{CSV_NAME}' found under: {PATIENT_DIR}")

    print(f"[INFO] Found {len(csv_files)} CSV(s) under {PATIENT_DIR}")
    summary_rows = []

    for csv_path in csv_files:
        visit_name, mosaic_name = parse_visit_and_mosaic(csv_path)
        n_classes = choose_n_classes(visit_name)

        # output folder per mosaic (keeps things organized)
        out_dir = OUT_ROOT / visit_name / mosaic_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {visit_name} | {mosaic_name} | {csv_path.name} | GMM={n_classes} ===")

        df = pd.read_csv(csv_path)

        # check columns
        missing = [c for c in (COL_G, COL_S) if c not in df.columns]
        if missing:
            print(f"  [WARN] Missing columns {missing}. Skipping.")
            continue

        G = safe_numeric(df[COL_G])
        S = safe_numeric(df[COL_S])
        valid = np.isfinite(G) & np.isfinite(S)

        Gv, Sv = G[valid], S[valid]
        X = np.column_stack([Gv, Sv])

        if X.shape[0] < 50:
            print(f"  [WARN] Too few valid phasor points (N={X.shape[0]}). Skipping.")
            continue

        # --- GMM segmentation ---
        gmm = GaussianMixture(
            n_components=int(n_classes),
            covariance_type=COV_TYPE,
            random_state=RANDOM_STATE,
        )
        labels_raw = gmm.fit_predict(X)

        # --- Auto-label ---
        if n_classes == 2:
            if COL_TAU not in df.columns:
                print(f"  [WARN] visit_04 requires '{COL_TAU}' for 2-cluster labeling. Skipping.")
                continue
            tau = safe_numeric(df[COL_TAU])[valid]
            label_map, centers_sorted, sort_mode = relabel_clusters_2_by_tau(labels_raw, tau)
            print("  Centers sorted by tau_phase_mean:")
            for lab, tau_c in centers_sorted:
                print(f"    raw {lab} | tau_phase_mean={tau_c:.4f} ns -> {label_map[lab]}")
        else:
            label_map, centers_sorted, sort_mode = relabel_clusters_3_by_angle(Gv, Sv, labels_raw)
            print("  Centers sorted by angle:")
            for lab, ang, gc, sc in centers_sorted:
                print(f"    raw {lab} | angle={ang:.4f} | (G,S)=({gc:.4f},{sc:.4f}) -> {label_map[lab]}")

        labels_name = np.array([label_map[int(l)] for l in labels_raw], dtype=object)

        counts = {
            "melanin": int(np.sum(labels_name == "melanin")) if n_classes == 3 else 0,
            "cell": int(np.sum(labels_name == "cell")),
            "elastin": int(np.sum(labels_name == "elastin")),
        }
        print("  Counts:", counts)

        # --- Write labels back (aligned to df rows) ---
        df["phasor_class_name"] = None
        df.loc[valid, "phasor_class_name"] = labels_name

        # --- Save classified CSV (all objects) ---
        out_all = out_dir / f"{csv_path.stem}_phasor_classified.csv"
        df.to_csv(out_all, index=False)

        # --- Save cells-only CSV ---
        df_cells = df[df["phasor_class_name"] == "cell"].copy()
        out_cells = out_dir / f"{csv_path.stem}_ONLY_cells.csv"
        df_cells.to_csv(out_cells, index=False)

        # --- Plot segmented phasor (only valid points) ---
        phasor_jpg = out_dir / f"{csv_path.stem}_phasor_segmented.jpg"
        title = f"{visit_name} | {mosaic_name} | GMM={n_classes} ({sort_mode})"
        plot = PhasorPlot(frequency=FREQUENCY_MHZ, title=title)

        classes_to_plot = ["cell", "elastin"] if n_classes == 2 else ["melanin", "cell", "elastin"]
        for cls in classes_to_plot:
            sel = labels_name == cls
            if np.sum(sel) == 0:
                continue
            plot.plot(
                Gv[sel], Sv[sel],
                marker=".",
                markersize=MARKER_SIZE,
                alpha=ALPHA,
                label=cls,
                color=CLASS_COLORS[cls],
            )

        plot.legend()
        save_phasorplot_high_quality(plot, phasor_jpg)

        # --- Summary row ---
        summary_rows.append({
            "visit": visit_name,
            "mosaic": mosaic_name,
            "csv_path": str(csv_path),
            "n_total_rows": int(len(df)),
            "n_valid_phasor": int(X.shape[0]),
            "n_classes_used": int(n_classes),
            "n_melanin": counts["melanin"],
            "n_cell": counts["cell"],
            "n_elastin": counts["elastin"],
            "classified_csv": str(out_all),
            "cells_only_csv": str(out_cells),
            "phasor_plot_jpg": str(phasor_jpg),
            "raw_label_map": str(label_map),
            "sort_mode": sort_mode,
        })

        print(f"  [SAVED] {out_dir}")

    # --- Save summary ---
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = OUT_ROOT / "summary_counts.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(f"\n✅ Done. Summary saved: {summary_csv}")
    print(f"📁 Outputs root: {OUT_ROOT}")


if __name__ == "__main__":
    main()