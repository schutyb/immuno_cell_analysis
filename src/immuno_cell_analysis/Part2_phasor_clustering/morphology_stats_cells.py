"""
morphology_stats_cells.py

Quantized violin plots per feature (multi-visit) + mean/median/modes vs visit
- Markers are ONLY short horizontal lines ("rayitas")
- Color coding:
    mean (promedio)  -> red
    median (mediana) -> green
    modes (modas)    -> blue (up to 3 modes via KDE peak detection)
- Labels include mini colored lines per parameter
- Also saves:
    mean vs visit (red)
    median vs visit (green)
    modes vs visit (blue; mode1/mode2/mode3)
- Saves ONLY JPG

This version auto-discovers the NEW pipeline files:
    *_ONLY_cells.csv
under:
    /Users/schutyb/Documents/balu_lab/data_patient_449/phasor_cluster_out
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.offsetbox import AnchoredOffsetbox, VPacker, HPacker, TextArea, DrawingArea
from matplotlib.lines import Line2D


# ============================================================
# CONFIG  ✅ CAMBIÁ SOLO ESTO SI MOVÉS LA CARPETA
# ============================================================
CELLS_CSV_ROOT = Path(
    "/Users/schutyb/Documents/balu_lab/data_patient_449/phasor_cluster_out"
)

# Where to save plots (inside the same root by default)
OUT_DIR = CELLS_CSV_ROOT / "plots_cells_violin_quantized"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Which columns to analyze (must exist in the ONLY_cells CSVs)
FEATURES = [
    "area_px2",
    "circularity",
    "equivalent_diameter_px",
    "major_axis_length_px",
    "minor_axis_length_px",
    "perimeter_px",
]

# If you want prettier labels:
# VISIT_LABELS = ["V1", "V2", "V3", "V4"]
VISIT_LABELS = None

# ---- Quantization step (choose based on effective resolution) ----
LEN_BIN = 0.25
AREA_BIN = 0.25
CIRC_BIN = 0.01
LENGTH_FEATURES = {"equivalent_diameter_px", "major_axis_length_px", "minor_axis_length_px", "perimeter_px"}

# ---- violin KDE (on quantized data) ----
BW_METHOD = 0.18
GRID_N = 2500
VIOLIN_WIDTH = 0.38

# ---- Mode finding (KDE peak detection) ----
MAX_MODES = 3
PEAK_MIN_REL_HEIGHT = 0.25   # peaks must be >= 25% of max density
PEAK_MIN_SEPARATION_BINS = 2 # separation in quantization bins

# ---- figure style ----
DPI = 160
FIG_W, FIG_H = 8.5, 4.8
FONT = 10

# ---- trend plots style ----
TREND_FIG_W, TREND_FIG_H = 6.5, 4.2

# ---- save (ONLY JPG) ----
EXPORT_DPI = 300
JPG_QUALITY = 95
JPG_SUBSAMPLING = 0

# ---- colors ----
C_MEAN = "red"
C_MED  = "green"
C_MODE = "blue"
C_EDGE = "0.35"
C_FILL = "0.85"
C_BLACK = "0.1"

# ---- label placement ----
LABEL_Y_TOP = 0.88
LABEL_Y_BOTTOM = 0.14


# ============================================================
# HELPERS
# ============================================================
def robust_numeric(series):
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return x[np.isfinite(x)]


def quantize_values(x: np.ndarray, feat: str) -> tuple[np.ndarray, float]:
    """Return quantized values + bin size used."""
    if x.size == 0:
        return x, 0.0
    if feat == "area_px2":
        bin_size = float(AREA_BIN)
    elif feat in LENGTH_FEATURES:
        bin_size = float(LEN_BIN)
    elif feat == "circularity":
        bin_size = float(CIRC_BIN)
    else:
        bin_size = 0.0

    if bin_size <= 0:
        return x, 0.0
    return (bin_size * np.round(x / bin_size)), bin_size


def local_maxima_indices(y: np.ndarray):
    if y.size < 3:
        return np.array([], dtype=int)
    return np.where((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]))[0] + 1


def find_modes_kde(xq: np.ndarray, bw_method: float, grid_n: int,
                   max_modes: int, min_rel_height: float, min_sep: float):
    if xq.size < 10:
        return []
    xmin, xmax = float(np.min(xq)), float(np.max(xq))
    if xmax <= xmin:
        return [xmin]

    kde = gaussian_kde(xq, bw_method=bw_method)
    grid = np.linspace(xmin, xmax, grid_n)
    dens = kde(grid)

    idx = local_maxima_indices(dens)
    if idx.size == 0:
        return [float(grid[int(np.argmax(dens))])]

    dmax = float(np.max(dens))
    keep = idx[dens[idx] >= (min_rel_height * dmax)]
    if keep.size == 0:
        return [float(grid[int(np.argmax(dens))])]

    keep = keep[np.argsort(dens[keep])[::-1]]  # by height desc

    chosen = []
    for i in keep:
        m = float(grid[int(i)])
        if all(abs(m - c) >= min_sep for c in chosen):
            chosen.append(m)
        if len(chosen) >= max_modes:
            break

    return sorted(chosen)


def stats_summary(xq: np.ndarray, modes: list[float]):
    q1, med, q3 = np.percentile(xq, [25, 50, 75])
    mean = float(np.mean(xq))
    p5, p95 = np.percentile(xq, [5, 95])
    return dict(
        n=int(len(xq)),
        mean=float(mean),
        med=float(med),
        q1=float(q1),
        q3=float(q3),
        p5=float(p5),
        p95=float(p95),
        modes=[float(m) for m in modes],
        iqr=float(q3 - q1),
    )


def fmt3(x):
    return f"{x:.3g}"


def matlab_like_violin(ax, x, x0, width, bw_method, grid_n):
    kde = gaussian_kde(x, bw_method=bw_method)
    y = np.linspace(np.min(x), np.max(x), grid_n)
    f = kde(y)
    fmax = np.max(f)
    if fmax > 0:
        f = (f / fmax) * width
    ax.fill(
        np.concatenate([x0 + f, (x0 - f)[::-1]]),
        np.concatenate([y, y[::-1]]),
        facecolor=C_FILL,
        edgecolor=C_EDGE,
        linewidth=1.0,
        alpha=1.0,
    )


def draw_iqr_whiskers(ax, x0, st, width):
    ax.add_patch(
        plt.Rectangle(
            (x0 - width*0.18, st["q1"]),
            width*0.36,
            st["q3"] - st["q1"],
            fill=False,
            linewidth=1.2,
            edgecolor=C_BLACK,
        )
    )
    ax.plot([x0, x0], [st["p5"], st["p95"]], linewidth=1.2, color=C_BLACK)
    ax.plot([x0 - width*0.12, x0 + width*0.12], [st["p5"], st["p5"]], linewidth=1.2, color=C_BLACK)
    ax.plot([x0 - width*0.12, x0 + width*0.12], [st["p95"], st["p95"]], linewidth=1.2, color=C_BLACK)


def ray(ax, x0, y, halfwidth, color, lw=2.6):
    ax.plot([x0 - halfwidth, x0 + halfwidth], [y, y],
            color=color, linewidth=lw, solid_capstyle="round")


def paper_axes(ax, ylabel, title, xticks, xticklabels):
    ax.set_title(title, fontsize=FONT + 2, pad=10)
    ax.set_ylabel(ylabel, fontsize=FONT)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=FONT)
    ax.tick_params(axis="y", labelsize=FONT)
    ax.grid(True, axis="y", alpha=0.25)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def save_jpg(fig, out_path: Path):
    fig.savefig(
        out_path,
        dpi=EXPORT_DPI,
        bbox_inches="tight",
        pil_kwargs={"quality": int(JPG_QUALITY), "subsampling": int(JPG_SUBSAMPLING)},
    )


def colored_line_legend_row(color, text, fontsize):
    da = DrawingArea(18, 10, 0, 0)
    line = Line2D([1, 17], [5, 5], color=color, linewidth=3, solid_capstyle="round")
    da.add_artist(line)
    ta = TextArea(text, textprops=dict(size=fontsize))
    return HPacker(children=[da, ta], align="center", pad=0, sep=3)


# ============================================================
# DISCOVER CSVs (NEW PIPELINE)
# ============================================================
def find_cells_csvs(root: Path) -> list[Path]:
    patterns = ["*_ONLY_cells.csv", "*ONLY_cells.csv"]
    hits = []
    for pat in patterns:
        hits.extend(root.rglob(pat))
    hits = sorted({p.resolve() for p in hits})
    return hits


def infer_visit_label(csv_path: Path) -> str:
    """
    Try to infer visit label from path like:
      .../phasor_cluster_out/visit_01/Mosaic.../*_ONLY_cells.csv
    """
    parts = list(csv_path.parts)
    for p in parts:
        if p.startswith("visit_"):
            return p
    # fallback: stem
    return csv_path.stem.replace("_ONLY_cells", "")


# ============================================================
# MAIN
# ============================================================
def main():
    if not CELLS_CSV_ROOT.exists():
        raise FileNotFoundError(f"CELLS_CSV_ROOT not found: {CELLS_CSV_ROOT}")

    csv_paths = find_cells_csvs(CELLS_CSV_ROOT)
    if not csv_paths:
        raise FileNotFoundError(f"No '*_ONLY_cells.csv' found under: {CELLS_CSV_ROOT}")

    print(f"[INFO] CELLS_CSV_ROOT: {CELLS_CSV_ROOT}")
    print(f"[INFO] Found {len(csv_paths)} ONLY_cells CSV(s):")
    for p in csv_paths:
        print("  ", p.relative_to(CELLS_CSV_ROOT))

    # Load & tag visits
    dfs = []
    visits = []
    for p in csv_paths:
        vlab = infer_visit_label(p)
        df = pd.read_csv(p)
        df["visit"] = vlab
        dfs.append(df)
        visits.append(vlab)

    # Unique visit order (sorted visit_01..)
    visits = sorted(set(visits), key=lambda x: int(x.split("_")[1]) if "_" in x and x.split("_")[1].isdigit() else x)

    if VISIT_LABELS is not None:
        if len(VISIT_LABELS) != len(visits):
            raise ValueError("VISIT_LABELS must have same length as number of visits discovered")
        visit_map = {v: lab for v, lab in zip(visits, VISIT_LABELS)}
    else:
        visit_map = {v: v for v in visits}

    all_df = pd.concat(dfs, ignore_index=True)

    print("\n[INFO] Loaded visits:")
    for v in visits:
        n = int(np.sum(all_df["visit"] == v))
        print(f"  {v} -> {visit_map[v]}: N={n}")

    # ============================================================
    # MAIN: per feature
    # ============================================================
    for feat in FEATURES:
        if feat not in all_df.columns:
            print(f"⚠️ Feature '{feat}' not found in CSVs. Skipping.")
            continue

        per_visit_q = []
        per_visit_st = []
        global_min = np.inf
        global_max = -np.inf

        mean_series = []
        median_series = []
        mode_series = [[], [], []]
        n_series = []

        for v in visits:
            x = robust_numeric(all_df.loc[all_df["visit"] == v, feat])
            xq, bin_size = quantize_values(x, feat)
            per_visit_q.append(xq)
            n_series.append(int(len(xq)))

            if xq.size < 10:
                per_visit_st.append(None)
                mean_series.append(np.nan)
                median_series.append(np.nan)
                for k in range(3):
                    mode_series[k].append(np.nan)
                continue

            xmin, xmax = float(np.min(xq)), float(np.max(xq))
            global_min = min(global_min, xmin)
            global_max = max(global_max, xmax)

            min_sep = (PEAK_MIN_SEPARATION_BINS * bin_size) if bin_size > 0 else (0.05 * (xmax - xmin))

            modes = find_modes_kde(
                xq,
                bw_method=BW_METHOD,
                grid_n=GRID_N,
                max_modes=MAX_MODES,
                min_rel_height=PEAK_MIN_REL_HEIGHT,
                min_sep=float(min_sep),
            )

            st = stats_summary(xq, modes=modes)
            per_visit_st.append(st)
            mean_series.append(st["mean"])
            median_series.append(st["med"])

            modes_pad = (modes + [np.nan]*3)[:3]
            for k in range(3):
                mode_series[k].append(modes_pad[k])

        if not np.isfinite(global_min) or not np.isfinite(global_max) or global_max <= global_min:
            print(f"⚠️ Too few usable data for '{feat}'. Skipping plots.")
            continue

        # ---------------------------
        # 1) Violin plot
        # ---------------------------
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
        x_positions = np.arange(1, len(visits) + 1, dtype=float)
        xticklabels = [visit_map[v] for v in visits]

        for i, (xq, st) in enumerate(zip(per_visit_q, per_visit_st), start=1):
            if st is None or xq.size < 10:
                continue

            matlab_like_violin(ax, xq, x0=float(i), width=VIOLIN_WIDTH, bw_method=BW_METHOD, grid_n=GRID_N)
            draw_iqr_whiskers(ax, x0=float(i), st=st, width=VIOLIN_WIDTH)

            hw = VIOLIN_WIDTH * 0.28
            ray(ax, float(i), st["mean"], halfwidth=hw, color=C_MEAN, lw=2.8)
            ray(ax, float(i), st["med"],  halfwidth=hw, color=C_MED,  lw=2.8)
            for m in st["modes"]:
                ray(ax, float(i), m, halfwidth=hw, color=C_MODE, lw=2.8)

            place = "bottom" if feat == "circularity" else "top"
            y_ax = LABEL_Y_BOTTOM if place == "bottom" else LABEL_Y_TOP

            mean_txt = f"mean={fmt3(st['mean'])}"
            med_txt  = f"median={fmt3(st['med'])}"
            modes_txt = ", ".join(fmt3(m) for m in st["modes"]) if st["modes"] else "NA"
            mode_txt = f"modes={modes_txt}"

            rows = [
                colored_line_legend_row(C_MEAN, mean_txt, fontsize=FONT-2),
                colored_line_legend_row(C_MED,  med_txt,  fontsize=FONT-2),
                colored_line_legend_row(C_MODE, mode_txt, fontsize=FONT-2),
            ]
            packed = VPacker(children=rows, align="left", pad=0, sep=2)

            box = AnchoredOffsetbox(
                loc="center",
                child=packed,
                pad=0.25,
                borderpad=0.35,
                frameon=True,
                bbox_to_anchor=(float(i), y_ax),
                bbox_transform=ax.get_xaxis_transform(),
            )
            box.patch.set_facecolor("white")
            box.patch.set_edgecolor("0.85")
            box.patch.set_alpha(0.92)
            ax.add_artist(box)

        title = f"{feat} (cells) — quantized violin by visit"
        paper_axes(ax, ylabel=feat, title=title, xticks=x_positions, xticklabels=xticklabels)

        pad = 0.22 * (global_max - global_min)
        ax.set_ylim(global_min - pad, global_max + pad)
        ax.set_xlim(0.5, len(visits) + 0.5)

        fig.tight_layout()
        save_jpg(fig, OUT_DIR / f"violin_{feat}_by_visit_quantized_multimode_rays.jpg")
        plt.close(fig)

        # ---------------------------
        # 2) Trend: mean
        # ---------------------------
        fig, ax = plt.subplots(figsize=(TREND_FIG_W, TREND_FIG_H), dpi=DPI)
        ax.plot(x_positions, mean_series, marker="o", color=C_MEAN)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(xticklabels, fontsize=FONT)
        ax.set_xlabel("Visit", fontsize=FONT)
        ax.set_ylabel(f"{feat} (mean)", fontsize=FONT)
        ax.set_title(f"{feat} — mean vs visit", fontsize=FONT + 2, pad=10)
        ax.grid(True, alpha=0.25)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for xi, yi, n in zip(x_positions, mean_series, n_series):
            if np.isfinite(yi):
                ax.text(xi, yi, f"N={n}", ha="center", va="bottom", fontsize=FONT-2)
        fig.tight_layout()
        save_jpg(fig, OUT_DIR / f"trend_mean_{feat}_vs_visit.jpg")
        plt.close(fig)

        # ---------------------------
        # 3) Trend: median
        # ---------------------------
        fig, ax = plt.subplots(figsize=(TREND_FIG_W, TREND_FIG_H), dpi=DPI)
        ax.plot(x_positions, median_series, marker="o", color=C_MED)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(xticklabels, fontsize=FONT)
        ax.set_xlabel("Visit", fontsize=FONT)
        ax.set_ylabel(f"{feat} (median)", fontsize=FONT)
        ax.set_title(f"{feat} — median vs visit", fontsize=FONT + 2, pad=10)
        ax.grid(True, alpha=0.25)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for xi, yi, n in zip(x_positions, median_series, n_series):
            if np.isfinite(yi):
                ax.text(xi, yi, f"N={n}", ha="center", va="bottom", fontsize=FONT-2)
        fig.tight_layout()
        save_jpg(fig, OUT_DIR / f"trend_median_{feat}_vs_visit.jpg")
        plt.close(fig)

        # ---------------------------
        # 4) Trend: modes (up to 3)
        # ---------------------------
        fig, ax = plt.subplots(figsize=(TREND_FIG_W, TREND_FIG_H), dpi=DPI)
        plotted_any = False
        for k in range(3):
            yk = np.array(mode_series[k], dtype=float)
            if np.any(np.isfinite(yk)):
                ax.plot(x_positions, yk, marker="o", color=C_MODE, alpha=0.9)
                plotted_any = True

        if plotted_any:
            ax.set_xticks(x_positions)
            ax.set_xticklabels(xticklabels, fontsize=FONT)
            ax.set_xlabel("Visit", fontsize=FONT)
            ax.set_ylabel(f"{feat} (modes)", fontsize=FONT)
            ax.set_title(f"{feat} — modes (up to 3) vs visit", fontsize=FONT + 2, pad=10)
            ax.grid(True, alpha=0.25)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            fig.tight_layout()
            save_jpg(fig, OUT_DIR / f"trend_modes_{feat}_vs_visit.jpg")
        plt.close(fig)

    print(f"\n✅ Done. Saved JPGs in: {OUT_DIR}")


if __name__ == "__main__":
    main()