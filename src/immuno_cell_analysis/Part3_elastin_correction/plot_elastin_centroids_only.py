#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from phasorpy.plot import PhasorPlot


# ============================================================
# CONFIG  👉 CAMBIÁ SOLO ESTO SI HACE FALTA
# ============================================================
# Donde están los CSV clasificados (Part2):
CLASSIFIED_ROOT = Path("/Users/schutyb/Documents/balu_lab/data_patient_449/phasor_cluster_out")

# Donde quedaron los corrected/calibrated (Part3):
CORRECTED_ROOT = CLASSIFIED_ROOT / "part3_elastin_phase_mod_correction_out"

OUT_DIR = CORRECTED_ROOT / "qc_elastin_centroids_only"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# patrones
CLASSIFIED_PATTERN = "*_phasor_classified.csv"
CORRECTED_PATTERN  = "*_phasor_classified_corrected.csv"  # usamos "corregido" (no final)

# columnas
CLASS_COL = "phasor_class_name"
ELASTIN_NAME = "elastin"

COL_G0 = "g_mean"
COL_S0 = "s_mean"

COL_GC = "g_mean_corr"
COL_SC = "s_mean_corr"

FREQUENCY_MHZ = 80.0

# estilo
MS = 3
ALPHA_VISIT = 1.0
ALPHA_GLOBAL = 1.0

COLOR_VISIT_ORIG = "blue"
COLOR_VISIT_CORR = "green"
COLOR_GLOBAL_ORIG = "orange"
COLOR_GLOBAL_CORR = "red"

# Zoom
USE_AUTO_ZOOM = True
ZOOM_PAD = 0.03
MANUAL_XLIM = (0.32, 0.45)
MANUAL_YLIM = (0.30, 0.42)

# Output
OUT_JPG = OUT_DIR / "phasor_elastin_centroids_original_vs_corrected_with_zoom.jpg"
EXPORT_DPI = 300
JPG_QUALITY = 95
JPG_SUBSAMPLING = 0


# ============================================================
# Helpers
# ============================================================
def safe_numeric(series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)

def finite_mask(G, S):
    return np.isfinite(G) & np.isfinite(S)

def centroid(G: np.ndarray, S: np.ndarray) -> tuple[float, float]:
    return float(np.mean(G)), float(np.mean(S))

def extract_visit_name_from_path(p: Path) -> str:
    for part in p.parts:
        if part.startswith("visit_"):
            return part
    return p.stem

def save_fig_jpg(fig, out_path: Path):
    fig.savefig(
        out_path,
        dpi=EXPORT_DPI,
        bbox_inches="tight",
        pil_kwargs={"quality": int(JPG_QUALITY), "subsampling": int(JPG_SUBSAMPLING)},
    )

def corrected_path_for_classified(classified_csv: Path) -> Path | None:
    """
    Busca el corrected correspondiente en CORRECTED_ROOT.
    Mantiene el mismo nombre base y sólo cambia el sufijo.
    """
    base = classified_csv.name.replace("_phasor_classified.csv", "_phasor_classified_corrected.csv")

    # 1) intento: misma estructura relativa (visit/mosaic/...)
    try:
        rel = classified_csv.relative_to(CLASSIFIED_ROOT)
        cand = (CORRECTED_ROOT / rel.parent / base)
        if cand.exists():
            return cand
    except Exception:
        pass

    # 2) fallback: búsqueda global por nombre exacto dentro de CORRECTED_ROOT
    hits = sorted(CORRECTED_ROOT.rglob(base))
    if hits:
        return hits[0]

    return None


# ============================================================
# Main
# ============================================================
def main():
    classified_files = sorted(CLASSIFIED_ROOT.rglob(CLASSIFIED_PATTERN))
    if not classified_files:
        raise FileNotFoundError(
            f"No classified CSVs found.\nRoot: {CLASSIFIED_ROOT}\nPattern: {CLASSIFIED_PATTERN}"
        )

    # agrupar por visita (visit_04 con 2 mosaicos se junta)
    per_visit = {}
    for fp in classified_files:
        visit = extract_visit_name_from_path(fp)
        per_visit.setdefault(visit, []).append(fp)

    visits = sorted(per_visit.keys())
    print(f"[INFO] Visits: {visits}")
    print(f"[INFO] CLASSIFIED_ROOT: {CLASSIFIED_ROOT}")
    print(f"[INFO] CORRECTED_ROOT:  {CORRECTED_ROOT}")

    rows = []
    visit_centroids_orig = []
    visit_centroids_corr = []

    for visit in visits:
        fps = per_visit[visit]

        G0_all, S0_all = [], []
        GC_all, SC_all = [], []

        for fp in fps:
            # -------- ORIGINAL --------
            df0 = pd.read_csv(fp)

            need0 = [CLASS_COL, COL_G0, COL_S0]
            if any(c not in df0.columns for c in need0):
                print(f"[WARN] {fp.name}: missing ORIGINAL columns -> skipping file")
                continue

            cls0 = df0[CLASS_COL].astype(str).str.lower().to_numpy()
            elast0 = (cls0 == ELASTIN_NAME)

            g0 = safe_numeric(df0[COL_G0])
            s0 = safe_numeric(df0[COL_S0])
            m0 = finite_mask(g0, s0) & elast0

            if np.any(m0):
                G0_all.append(g0[m0])
                S0_all.append(s0[m0])

            # -------- CORRECTED --------
            fp_corr = corrected_path_for_classified(fp)
            if fp_corr is None:
                print(f"[WARN] Missing corrected CSV for: {fp}  (searched in CORRECTED_ROOT)")
                continue

            dfc = pd.read_csv(fp_corr)
            needc = [CLASS_COL, COL_GC, COL_SC]
            if any(c not in dfc.columns for c in needc):
                print(f"[WARN] {fp_corr.name}: missing CORRECTED columns -> skipping corrected file")
                continue

            clsc = dfc[CLASS_COL].astype(str).str.lower().to_numpy()
            elastc = (clsc == ELASTIN_NAME)

            gc = safe_numeric(dfc[COL_GC])
            sc = safe_numeric(dfc[COL_SC])
            mc = finite_mask(gc, sc) & elastc

            if np.any(mc):
                GC_all.append(gc[mc])
                SC_all.append(sc[mc])

        if len(G0_all) == 0 or len(GC_all) == 0:
            print(f"[WARN] Visit {visit}: insufficient elastin points (orig or corr). Skipping visit.")
            continue

        G0v = np.concatenate(G0_all)
        S0v = np.concatenate(S0_all)
        GCv = np.concatenate(GC_all)
        SCv = np.concatenate(SC_all)

        g0c, s0c = centroid(G0v, S0v)
        gcc, scc = centroid(GCv, SCv)

        visit_centroids_orig.append((g0c, s0c))
        visit_centroids_corr.append((gcc, scc))

        rows.append({
            "visit": visit,
            "N_elastin_orig": int(len(G0v)),
            "N_elastin_corr": int(len(GCv)),
            "G_orig": g0c, "S_orig": s0c,
            "G_corr": gcc, "S_corr": scc,
        })

    if not rows:
        raise RuntimeError(
            "No per-visit elastin centroids computed.\n"
            "Most likely: CORRECTED_ROOT is wrong or corrected files were not generated."
        )

    df_cent = pd.DataFrame(rows).sort_values("visit").reset_index(drop=True)

    # “centro de masa de los centros de masa”
    g0m = float(np.mean([x for x, _ in visit_centroids_orig]))
    s0m = float(np.mean([y for _, y in visit_centroids_orig]))
    gcm = float(np.mean([x for x, _ in visit_centroids_corr]))
    scm = float(np.mean([y for _, y in visit_centroids_corr]))

    # Save table
    out_csv = OUT_DIR / "elastin_centroids_only_orig_vs_corr_plus_global.csv"
    df_cent.assign(
        global_G_orig_mean=g0m,
        global_S_orig_mean=s0m,
        global_G_corr_mean=gcm,
        global_S_corr_mean=scm,
    ).to_csv(out_csv, index=False)

    print(f"[SAVED] {out_csv}")

    # Zoom window
    if USE_AUTO_ZOOM:
        xlim = (gcm - ZOOM_PAD, gcm + ZOOM_PAD)
        ylim = (scm - ZOOM_PAD, scm + ZOOM_PAD)
    else:
        xlim = MANUAL_XLIM
        ylim = MANUAL_YLIM

    # Plot: full + zoom
    fig = plt.figure(figsize=(14, 6), dpi=200)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    pp1 = PhasorPlot(ax=ax1, frequency=FREQUENCY_MHZ, title="Elastin centroids — FULL")
    pp2 = PhasorPlot(ax=ax2, frequency=FREQUENCY_MHZ, title="Elastin centroids — ZOOM")

    for _, r in df_cent.iterrows():
        v = r["visit"]

        # original (blue)
        pp1.plot([r["G_orig"]], [r["S_orig"]], marker="o", markersize=MS,
                 alpha=ALPHA_VISIT, color=COLOR_VISIT_ORIG, label=f"{v} orig")
        pp2.plot([r["G_orig"]], [r["S_orig"]], marker="o", markersize=MS,
                 alpha=ALPHA_VISIT, color=COLOR_VISIT_ORIG, label=f"{v} orig")

        # corrected (green) → un poco más grandes
        pp1.plot([r["G_corr"]], [r["S_corr"]],
                marker="o", markersize=5,
                alpha=ALPHA_VISIT, color=COLOR_VISIT_CORR,
                label=f"{v} corr")

        pp2.plot([r["G_corr"]], [r["S_corr"]],
                marker="o", markersize=5,
                alpha=ALPHA_VISIT, color=COLOR_VISIT_CORR,
                label=f"{v} corr")

    # global means
    pp1.plot([g0m], [s0m], marker="o", markersize=MS, alpha=ALPHA_GLOBAL,
             color=COLOR_GLOBAL_ORIG, label="GLOBAL mean orig")
    pp2.plot([g0m], [s0m], marker="o", markersize=MS, alpha=ALPHA_GLOBAL,
             color=COLOR_GLOBAL_ORIG, label="GLOBAL mean orig")

    pp1.plot([gcm], [scm], marker="o", markersize=MS, alpha=ALPHA_GLOBAL,
             color=COLOR_GLOBAL_CORR, label="GLOBAL mean corr")
    pp2.plot([gcm], [scm], marker="o", markersize=MS, alpha=ALPHA_GLOBAL,
             color=COLOR_GLOBAL_CORR, label="GLOBAL mean corr")

    # zoom
    ax2.set_xlim(*xlim)
    ax2.set_ylim(*ylim)

    ax1.legend(loc="best", fontsize=8, frameon=True)

    fig.suptitle(
        "Elastin centroids per visit: ORIGINAL (blue) vs CORRECTED (green) + global means (orange/red)",
        y=0.98
    )
    fig.tight_layout()

    save_fig_jpg(fig, OUT_JPG)
    plt.close(fig)

    print(f"[SAVED] {OUT_JPG}")
    print(f"[INFO] Zoom xlim={xlim}, ylim={ylim}")
    print(f"✅ Done. Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()

