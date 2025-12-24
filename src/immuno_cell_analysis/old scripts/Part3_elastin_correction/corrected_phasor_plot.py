from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from phasorpy.plot import PhasorPlot


# ============================================================
# CONFIG
# ============================================================
INPUT_ROOT = Path(
    "/Users/schutyb/Documents/balu_lab/data_patient_449/"
    "phasor_cluster_out/part3_elastin_phase_mod_correction_out"
)

OUT_DIR = INPUT_ROOT / "qc_phasor_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATTERN = "*_phasor_classified_corrected_calibrated.csv"
FREQUENCY_MHZ = 80.0

CLASS_COL = "phasor_class_name"

# ORIGINAL
G0, S0 = "g_mean", "s_mean"

# FINAL
GF, SF = "g_mean_final", "s_mean_final"

# ---- Plot style ----
MARKER = "."
MARKERSIZE = 4
ALPHA = 1.0

# ---- Colors ----
CLASS_COLORS_ORIG = {
    "elastin": "#2ca02c",  # green
    "cell":    "#ff7f0e",  # orange
    "melanin": "#1f77b4",  # blue
}

CLASS_COLORS_FINAL = {
    "elastin": "#7f7f7f",  # gray
    "cell":    "#d62728",  # red
    "melanin": "#9467bd",  # lilac
}

# ---- Export ----
EXPORT_DPI = 600
JPG_QUALITY = 95
JPG_SUBSAMPLING = 0


# ============================================================
# Helpers
# ============================================================
def safe_numeric(series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def finite_mask(G: np.ndarray, S: np.ndarray) -> np.ndarray:
    return np.isfinite(G) & np.isfinite(S)


def find_csvs(root: Path) -> list[Path]:
    return sorted(root.rglob(PATTERN))


def visit_from_path(p: Path) -> str:
    for part in p.parts:
        if part.startswith("visit_"):
            return part
    return "visit_unknown"


def save_phasorplot_high_quality(plot_obj, out_path: Path):
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
            pil_kwargs={"quality": JPG_QUALITY, "subsampling": JPG_SUBSAMPLING},
        )
        return

    plot_obj.save(str(out_path))


def close_plot(plot_obj):
    fig = getattr(plot_obj, "fig", None) or getattr(plot_obj, "figure", None)
    if fig is not None:
        plt.close(fig)


def add_points_by_class(plot, G, S, cls, label_prefix, class_colors):
    m = finite_mask(G, S)
    if m.sum() == 0:
        return

    Gv, Sv, clsv = G[m], S[m], cls[m]

    for cname in ["melanin", "cell", "elastin"]:
        sel = clsv == cname
        if np.sum(sel) == 0:
            continue

        plot.plot(
            Gv[sel],
            Sv[sel],
            marker=MARKER,
            markersize=MARKERSIZE,
            alpha=ALPHA,
            color=class_colors[cname],
            label=f"{label_prefix} | {cname}",
        )


# ============================================================
# Main
# ============================================================
def main():
    print(f"[INFO] INPUT_ROOT: {INPUT_ROOT}")
    print(f"[INFO] OUT_DIR:    {OUT_DIR}")

    csvs = find_csvs(INPUT_ROOT)
    if not csvs:
        raise FileNotFoundError("No calibrated CSVs found.")

    per_visit: dict[str, list[pd.DataFrame]] = {}
    for p in csvs:
        df = pd.read_csv(p)

        needed = [CLASS_COL, G0, S0, GF, SF]
        if any(c not in df.columns for c in needed):
            print(f"[WARN] {p.name} missing required columns → skipped")
            continue

        v = visit_from_path(p)
        per_visit.setdefault(v, []).append(df)

    visits = sorted(per_visit.keys())
    print(f"[INFO] Visits: {visits}")

    # --------------------------------------------------------
    # Per-visit ORIG vs FINAL
    # --------------------------------------------------------
    for v in visits:
        dfv = pd.concat(per_visit[v], ignore_index=True)
        cls = dfv[CLASS_COL].astype(str).str.lower().to_numpy()

        g0, s0 = safe_numeric(dfv[G0]), safe_numeric(dfv[S0])
        gf, sf = safe_numeric(dfv[GF]), safe_numeric(dfv[SF])

        plot = PhasorPlot(
            frequency=FREQUENCY_MHZ,
            title=f"{v} — Phasor QC (ORIG vs FINAL)",
        )

        add_points_by_class(plot, g0, s0, cls, "ORIG",  CLASS_COLORS_ORIG)
        add_points_by_class(plot, gf, sf, cls, "FINAL", CLASS_COLORS_FINAL)

        plot.legend()
        out = OUT_DIR / f"{v}_phasor_ORIG_vs_FINAL_600dpi.jpg"
        save_phasorplot_high_quality(plot, out)
        close_plot(plot)
        print(f"[SAVED] {out.name}")

    # --------------------------------------------------------
    # Global FINAL plot
    # --------------------------------------------------------
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    visit_colors = {v: palette[i % len(palette)] for i, v in enumerate(visits)}

    plot_all = PhasorPlot(
        frequency=FREQUENCY_MHZ,
        title="FINAL calibrated phasors — all visits",
    )

    for v in visits:
        dfv = pd.concat(per_visit[v], ignore_index=True)
        gf, sf = safe_numeric(dfv[GF]), safe_numeric(dfv[SF])
        m = finite_mask(gf, sf)
        if m.sum() < 20:
            continue

        plot_all.plot(
            gf[m],
            sf[m],
            marker=MARKER,
            markersize=MARKERSIZE,
            alpha=ALPHA,
            color=visit_colors[v],
            label=v,
        )

    plot_all.legend()
    out_all = OUT_DIR / "ALL_visits_FINAL_600dpi.jpg"
    save_phasorplot_high_quality(plot_all, out_all)
    close_plot(plot_all)

    print(f"[SAVED] {out_all.name}")
    print(f"\n✅ Done. Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()