#!/usr/bin/env python3
"""
Plot phasor clouds from autocalibrated (GREEN-only) per-structure CSVs.

- Reads CSVs from a folder (glob)
- Uses green_g_mean and green_s_mean
- Auto-detect class column: phasor_class_name OR phasor_class
- Colors by class:
    elastin -> green
    cell    -> orange
    melanin -> blue
- Saves one plot per CSV + one combined plot.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from phasorpy.plot import PhasorPlot


# ============================================================
# CONFIG (EDIT ME)
# ============================================================
INPUT_DIR = Path("/Users/schutyb/Documents/balu_lab/immuno_cell_analysis_data/features_table/elastin_corrected_out/")  # folder with *_elastinAutocal_GREEN.csv
CSV_GLOB = "*_elastinAutocal_GREEN.csv"

COL_G = "green_g_mean"
COL_S = "green_s_mean"

# class column candidates (we'll pick the first found)
CLASS_COL_CANDIDATES = ["phasor_class_name", "phasor_class"]

MAX_POINTS_PER_CLASS = 150_000  # set None to disable

FREQUENCY_MHZ = 80.0
MARKER_SIZE = 7
ALPHA = 0.45
DPI = 300

OUT_DIR = None  # None => INPUT_DIR / "phasor_plots"

CLASS_COLORS = {
    "elastin": "green",
    "cell": "orange",
    "melanin": "blue",
}
# ============================================================


def safe_float(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def normalize_class_name(x) -> str:
    if x is None:
        return ""
    return str(x).strip().lower()


def subsample(G: np.ndarray, S: np.ndarray, max_points: int | None, seed: int = 0):
    if max_points is None or G.size <= max_points:
        return G, S
    rng = np.random.default_rng(seed)
    idx = rng.choice(G.size, size=max_points, replace=False)
    return G[idx], S[idx]


def ensure_out_dir() -> Path:
    out = OUT_DIR if OUT_DIR is not None else (INPUT_DIR / "phasor_plots")
    out.mkdir(parents=True, exist_ok=True)
    return out


def find_class_col(df: pd.DataFrame) -> str | None:
    for c in CLASS_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def plot_one_csv(csv_path: Path, out_dir: Path):
    df = pd.read_csv(csv_path)

    class_col = find_class_col(df)
    missing = [c for c in (COL_G, COL_S) if c not in df.columns]
    if missing or class_col is None:
        print(f"[WARN] {csv_path.name}: missing columns {missing + (['class_col'] if class_col is None else [])}. Skipping.")
        print(f"       Available columns (first 40): {list(df.columns)[:40]}")
        return None

    G = safe_float(df[COL_G])
    S = safe_float(df[COL_S])
    cls = df[class_col].map(normalize_class_name).to_numpy(dtype=object)

    valid = np.isfinite(G) & np.isfinite(S) & (cls != "")
    if valid.sum() == 0:
        print(f"[WARN] {csv_path.name}: no valid phasor points. Skipping.")
        return None

    Gv, Sv, Cv = G[valid], S[valid], cls[valid]

    title = f"{csv_path.stem}"
    plot = PhasorPlot(frequency=FREQUENCY_MHZ, title=title)

    for cname in ["elastin", "cell", "melanin"]:
        sel = Cv == cname
        if np.sum(sel) == 0:
            continue

        g_i, s_i = subsample(Gv[sel], Sv[sel], MAX_POINTS_PER_CLASS, seed=0)
        plot.plot(
            g_i, s_i,
            marker=".",
            markersize=MARKER_SIZE,
            alpha=ALPHA,
            label=f"{cname} (N={np.sum(sel):,})",
            color=CLASS_COLORS[cname],
        )

    plot.semicircle()
    plot.legend()

    out_png = out_dir / f"{csv_path.stem}_phasor.png"
    try:
        plot.save(str(out_png), dpi=DPI)
    except Exception:
        fig = getattr(plot, "fig", None) or getattr(plot, "figure", None)
        if fig is not None:
            fig.savefig(str(out_png), dpi=DPI, bbox_inches="tight")
        else:
            plot.save(str(out_png))

    print(f"[OK] Saved: {out_png}")
    return (Gv, Sv, Cv)


def plot_combined(all_data, out_dir: Path):
    if not all_data:
        return

    plot = PhasorPlot(frequency=FREQUENCY_MHZ, title="All visits (autocalibrated GREEN)")

    G_all = np.concatenate([d[0] for d in all_data])
    S_all = np.concatenate([d[1] for d in all_data])
    C_all = np.concatenate([d[2] for d in all_data])

    for cname in ["elastin", "cell", "melanin"]:
        sel = C_all == cname
        if np.sum(sel) == 0:
            continue

        g_i, s_i = subsample(G_all[sel], S_all[sel], MAX_POINTS_PER_CLASS, seed=1)
        plot.plot(
            g_i, s_i,
            marker=".",
            markersize=MARKER_SIZE,
            alpha=ALPHA,
            label=f"{cname} (N={np.sum(sel):,})",
            color=CLASS_COLORS[cname],
        )

    plot.semicircle()
    plot.legend()

    out_png = out_dir / "ALL_visits_phasor.png"
    try:
        plot.save(str(out_png), dpi=DPI)
    except Exception:
        fig = getattr(plot, "fig", None) or getattr(plot, "figure", None)
        if fig is not None:
            fig.savefig(str(out_png), dpi=DPI, bbox_inches="tight")
        else:
            plot.save(str(out_png))

    print(f"[OK] Saved combined: {out_png}")


def main():
    out_dir = ensure_out_dir()
    csvs = sorted(INPUT_DIR.glob(CSV_GLOB))
    if not csvs:
        raise FileNotFoundError(f"No CSVs found with glob '{CSV_GLOB}' under {INPUT_DIR}")

    print(f"[INFO] Found {len(csvs)} CSV(s)")

    all_data = []
    for p in csvs:
        res = plot_one_csv(p, out_dir)
        if res is not None:
            all_data.append(res)

    plot_combined(all_data, out_dir)
    print("\n✅ Done.")


if __name__ == "__main__":
    main()