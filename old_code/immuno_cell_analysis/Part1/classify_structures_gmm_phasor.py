#!/usr/bin/env python3
"""
GMM clustering on per-structure mean phasor (GREEN) and add a class label column to CSV.

Clustering features:
  - green_g_mean
  - green_s_mean

Auto-labeling rule (by phasor angle atan2(S, G)):
  - 3 clusters: smallest angle -> melanin, middle -> cell, largest -> elastin
  - 2 clusters: (default) lower angle -> cell, higher angle -> elastin
    (optional) melanin_elastin mode: lower -> melanin, higher -> elastin

Outputs:
  - <input_stem>_phasor_classified.csv
  - <input_stem>_phasor_segmented.png (optional)

Edit CONFIG and run:
  python classify_structures_gmm_phasor.py
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
IN_CSV = Path(
    "/Users/schutyb/Documents/balu_lab/immuno_cell_analysis_data/visit_04/Mosaic07_4x4_FOV600_z150_32Sp/structure_features.csv")

# Columns in your features CSV:
COL_ID = "structure_id"
COL_G = "green_g_mean"
COL_S = "green_s_mean"

# Choose number of clusters: 2 or 3
N_CLASSES = 2

# For N_CLASSES == 2, choose labeling mode:
#   "cell_elastin" (default, assumes melanin absent)
#   "melanin_elastin" (if cell absent)
TWO_CLASS_MODE = "cell_elastin"

# GMM settings
RANDOM_STATE = 0
COV_TYPE = "full"

# Output
SAVE_PLOT = True
PLOT_FILENAME = None  # None => auto: "<stem>_phasor_segmented.png"
PLOT_FREQUENCY_MHZ = 80.0
PLOT_MARKER_SIZE = 8
PLOT_ALPHA = 0.55
PLOT_DPI = 300

# Fixed class colors (optional; nice for consistency)
CLASS_COLORS = {
    "melanin": "#1f77b4",
    "cell":    "#ff7f0e",
    "elastin": "#2ca02c",
}
# ============================================================


def _safe_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _cluster_centers_by_angle(G: np.ndarray, S: np.ndarray, labels_raw: np.ndarray):
    """Return list of tuples: (raw_label, angle, Gc, Sc) sorted by angle."""
    centers = []
    for lab in np.unique(labels_raw):
        sel = labels_raw == lab
        Gc = float(np.mean(G[sel]))
        Sc = float(np.mean(S[sel]))
        ang = float(np.arctan2(Sc, Gc))
        centers.append((int(lab), ang, Gc, Sc))
    centers.sort(key=lambda t: t[1])
    return centers


def _make_label_map(centers_sorted, n_classes: int, two_class_mode: str):
    """
    Map raw cluster labels -> class names based on sorted angle order.
    """
    if n_classes == 3:
        ordered_names = ["melanin", "cell", "elastin"]  # low -> high angle
    elif n_classes == 2:
        if two_class_mode == "melanin_elastin":
            ordered_names = ["melanin", "elastin"]
        else:
            # default and recommended in your pipeline (melanin absent)
            ordered_names = ["cell", "elastin"]
    else:
        raise ValueError("n_classes must be 2 or 3")

    if len(centers_sorted) != len(ordered_names):
        raise RuntimeError("Unexpected number of clusters returned by GMM.")

    return {raw_lab: name for (raw_lab, *_), name in zip(centers_sorted, ordered_names)}


def _insert_column_after(df: pd.DataFrame, after_col: str, new_col: str, values) -> pd.DataFrame:
    """Insert new_col right after after_col; if after_col missing, append at end."""
    df2 = df.copy()
    if new_col in df2.columns:
        df2 = df2.drop(columns=[new_col])

    if after_col in df2.columns:
        idx = df2.columns.get_loc(after_col) + 1
        df2.insert(idx, new_col, values)
    else:
        df2[new_col] = values
    return df2


def _save_segmented_phasor_plot(out_path: Path, G: np.ndarray, S: np.ndarray, cls: np.ndarray, n_classes: int):
    title = f"GMM={n_classes} on ({COL_G}, {COL_S})"
    plot = PhasorPlot(frequency=PLOT_FREQUENCY_MHZ, title=title)

    classes_to_plot = ["cell", "elastin"] if n_classes == 2 else ["melanin", "cell", "elastin"]
    for cname in classes_to_plot:
        sel = cls == cname
        if np.sum(sel) == 0:
            continue
        plot.plot(
            G[sel], S[sel],
            marker=".",
            markersize=PLOT_MARKER_SIZE,
            alpha=PLOT_ALPHA,
            label=cname,
            color=CLASS_COLORS.get(cname, None),
        )

    plot.legend()

    # Save robustly
    try:
        plot.save(str(out_path), dpi=PLOT_DPI)
        return
    except Exception:
        pass

    fig = getattr(plot, "fig", None) or getattr(plot, "figure", None)
    if fig is not None:
        fig.savefig(str(out_path), dpi=PLOT_DPI, bbox_inches="tight")
        return

    # last fallback
    plot.save(str(out_path))


def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(f"CSV not found: {IN_CSV}")

    if N_CLASSES not in (2, 3):
        raise ValueError("N_CLASSES must be 2 or 3")

    df = pd.read_csv(IN_CSV)

    for col in (COL_ID, COL_G, COL_S):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in CSV.")

    G = _safe_numeric(df[COL_G])
    S = _safe_numeric(df[COL_S])
    valid = np.isfinite(G) & np.isfinite(S)

    if int(valid.sum()) < max(30, 10 * N_CLASSES):
        raise ValueError(f"Too few valid phasor points: N_valid={int(valid.sum())}")

    X = np.column_stack([G[valid], S[valid]])

    gmm = GaussianMixture(
        n_components=int(N_CLASSES),
        covariance_type=COV_TYPE,
        random_state=RANDOM_STATE,
    )
    labels_raw = gmm.fit_predict(X)

    centers_sorted = _cluster_centers_by_angle(G[valid], S[valid], labels_raw)
    label_map = _make_label_map(centers_sorted, N_CLASSES, TWO_CLASS_MODE)

    # human-readable labels
    labels_name = np.array([label_map[int(l)] for l in labels_raw], dtype=object)

    # create full column aligned with df
    phasor_class_full = np.full(len(df), None, dtype=object)
    phasor_class_full[valid] = labels_name

    # insert near ID
    df_out = _insert_column_after(df, COL_ID, "phasor_class", phasor_class_full)

    out_csv = IN_CSV.with_name(f"{IN_CSV.stem}_phasor_classified.csv")
    df_out.to_csv(out_csv, index=False)

    print("✔ Classified CSV saved:")
    print(f"  {out_csv}")

    print("Cluster centers (sorted by angle atan2(S,G)):")
    for raw_lab, ang, gc, sc in centers_sorted:
        print(f"  raw {raw_lab:>2} | angle={ang:+.4f} | (G,S)=({gc:.4f},{sc:.4f}) -> {label_map[raw_lab]}")

    if SAVE_PLOT:
        if PLOT_FILENAME is None:
            out_plot = IN_CSV.with_name(f"{IN_CSV.stem}_phasor_segmented.png")
        else:
            out_plot = IN_CSV.with_name(PLOT_FILENAME)

        _save_segmented_phasor_plot(out_plot, G[valid], S[valid], labels_name, N_CLASSES)
        print("✔ Phasor plot saved:")
        print(f"  {out_plot}")


if __name__ == "__main__":
    main()