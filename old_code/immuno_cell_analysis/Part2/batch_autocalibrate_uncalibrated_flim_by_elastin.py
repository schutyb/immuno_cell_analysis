#!/usr/bin/env python3
"""
Batch autocalibration of UNCALIBRATED FLIM mosaics using elastin,
with GLOBAL reference derived from Coumarin-calibrated CSVs.

Folder structure expected (Option A):
ROOT/
  visit_01/
    MosaicXX.../
      flim_mosaic.tif                          # UNCALIBRATED raw FLIM
      elastin_mask_mosaic.png                  # elastin mask (binary)
      structure_features_phasor_classified.csv # Coumarin-calibrated per-structure table
  visit_02/...
  visit_03/...
  visit_04/...

Pipeline:
A) Build GLOBAL_REF elastin stats (calibrated) from the 4 CSVs (elastin only):
   mu_phi_bar, sig_phi_bar, mu_r_bar, sig_r_bar

B) For each visit/mosaic:
   1) From UNCALIBRATED flim_mosaic.tif, compute per-pixel phasor (GREEN only).
   2) Use elastin mask to compute per-visit stats (uncalibrated):
        mu_phi_v, sig_phi_v, mu_r_v, sig_r_v
   3) Apply whiteboard autocalibration:

      # standardized mapping
      dphi = wrap(phi - mu_phi_v)
      phi1 = wrap((dphi/sig_phi_v)*sig_phi_bar + mu_phi_bar)
      r1   = ((r-mu_r_v)/sig_r_v)*sig_r_bar + mu_r_bar

      # compute final correction from elastin pixels
      delta_phi = wrap(mu_phi_bar - mean_circular(phi1_elastin))
      Fr        = mu_r_bar / mean(r1_elastin)

      # apply to ORIGINAL raw (phi,r)
      phi2 = wrap(phi + delta_phi)
      r2   = Fr * r

   4) Save phasor_uncalibrated_autocal_GREEN_CYX.tif (CYX planes).
   5) Save QC plot: elastin phasor before vs after (per mosaic + combined),
      using PhasorPlot so the universal semicircle is shown.

Outputs:
ROOT/autocalibration_out/
  elastin_reference_calibrated_GLOBAL.csv
  per_visit_uncal_elastin_stats.csv
  visit_XX/MosaicYY/phasor_uncalibrated_autocal_GREEN_CYX.tif
  visit_XX/MosaicYY/elastin_phasor_before_after.png
  ALL_elastin_before_after.png
"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import tifffile as tiff

from phasorpy.phasor import phasor_from_signal, phasor_to_polar
from phasorpy.lifetime import polar_to_apparent_lifetime
from phasorpy.plot import PhasorPlot

import matplotlib.pyplot as plt


# ============================================================
# CONFIG (EDIT ME)
# ============================================================
ROOT = Path("/Users/schutyb/Documents/balu_lab/immuno_cell_analysis_data")

# Required filenames inside each Mosaic folder
FNAME_FLIM = "flim_mosaic.tif"  # uncalibrated
FNAME_MASK = "elastin_mask_mosaic.png"  # use as elastin mask
FNAME_CSV_CAL = "structure_features_phasor_classified.csv"  # calibrated (Coumarin)

# GREEN bins
GREEN_N_BINS = 16

# Use only pixels with mean intensity > threshold (avoid background)
INTENSITY_MIN = 1.0

# Class column candidates in CSV
CLASS_COL_CANDIDATES = ["phasor_class_name", "phasor_class"]
ELASTIN_NAME = "elastin"

# Which calibrated phasor columns exist in CSV? (we can compute polar from g/s)
COL_G_CAL = "green_g_mean"
COL_S_CAL = "green_s_mean"
COL_PHASE_CAL = "green_phase_mean"
COL_MOD_CAL = "green_modulation_mean"

# Output root
OUT_ROOT = ROOT / "autocalibration_out"

# Save lifetimes for the final corrected polar
FREQUENCY_MHZ = 80.0
UNIT_CONVERSION = 1e-3  # MHz + ns

# Performance / QC plotting
MAX_PIXELS_FOR_STATS = 2_000_000      # subsample elastin pixels for per-visit stats
MAX_POINTS_FOR_QC_SCATTER = 200_000   # subsample elastin pixels for plotting
RNG_SEED = 0

# Clip modulation after correction (optional safety)
CLIP_MOD_TO_01 = True

# PhasorPlot saving quality
EXPORT_DPI = 300
# ============================================================


# -------------------------
# Utilities
# -------------------------
def wrap_phase(x: np.ndarray) -> np.ndarray:
    """Wrap to [-pi, pi]."""
    return np.angle(np.exp(1j * x))


def circular_mean(a: np.ndarray) -> float:
    return float(np.arctan2(np.nanmean(np.sin(a)), np.nanmean(np.cos(a))))


def circular_std(a: np.ndarray) -> float:
    """
    Circular standard deviation (radians) using mean resultant length R:
      sigma = sqrt(-2 ln R)
    """
    s = np.nanmean(np.sin(a))
    c = np.nanmean(np.cos(a))
    R = float(np.clip(np.sqrt(s * s + c * c), 1e-12, 1.0))
    return float(np.sqrt(-2.0 * np.log(R)))


def subsample_1d(x: np.ndarray, max_n: int | None, seed: int) -> np.ndarray:
    if max_n is None or x.size <= max_n:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.size, size=max_n, replace=False)
    return x[idx]


def load_mask_2d(path: Path) -> np.ndarray:
    if path.suffix.lower() in [".tif", ".tiff"]:
        m = tiff.imread(str(path))
    else:
        from PIL import Image
        m = np.array(Image.open(str(path)))
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0)


def read_tiff_with_axes(path: Path) -> tuple[np.ndarray, str | None]:
    with tiff.TiffFile(str(path)) as tf:
        arr = tf.asarray()
        desc = tf.pages[0].description
    axes = None
    if desc:
        try:
            d = json.loads(desc.strip())
            axes = d.get("axes", None)
        except Exception:
            axes = None
    return np.asarray(arr), axes


def to_cyx(arr: np.ndarray, axes: str | None) -> np.ndarray:
    """
    Convert common FLIM shapes to CYX.
    Supports axes metadata: CYX or TYX (treat T as C).
    """
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D FLIM, got shape {arr.shape}")

    if axes:
        ax = axes.upper()
        if "C" in ax and "Y" in ax and "X" in ax:
            c = ax.index("C"); y = ax.index("Y"); x = ax.index("X")
            return np.moveaxis(arr, (c, y, x), (0, 1, 2))
        if "T" in ax and "Y" in ax and "X" in ax:
            t = ax.index("T"); y = ax.index("Y"); x = ax.index("X")
            return np.moveaxis(arr, (t, y, x), (0, 1, 2))

    # Heuristic: assume already (C,Y,X)
    return arr


def find_class_col(df: pd.DataFrame) -> str:
    for c in CLASS_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"No class column found in calibrated CSV. Tried: {CLASS_COL_CANDIDATES}")


def get_phase_mod_from_calibrated_csv(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Return phase, modulation for GREEN from calibrated per-structure CSV.
    Prefer phase/mod columns if present; else compute from g/s.
    """
    if COL_PHASE_CAL in df.columns and COL_MOD_CAL in df.columns:
        ph = pd.to_numeric(df[COL_PHASE_CAL], errors="coerce").to_numpy(dtype=float)
        r = pd.to_numeric(df[COL_MOD_CAL], errors="coerce").to_numpy(dtype=float)
        return ph, r

    if COL_G_CAL not in df.columns or COL_S_CAL not in df.columns:
        raise ValueError(
            f"Need either ({COL_PHASE_CAL},{COL_MOD_CAL}) or ({COL_G_CAL},{COL_S_CAL}) in calibrated CSV."
        )
    g = pd.to_numeric(df[COL_G_CAL], errors="coerce").to_numpy(dtype=float)
    s = pd.to_numeric(df[COL_S_CAL], errors="coerce").to_numpy(dtype=float)
    ph = np.arctan2(s, g)
    r = np.sqrt(g * g + s * s)
    return ph, r


# -------------------------
# Discovery
# -------------------------
def discover_jobs(root: Path) -> list[dict]:
    """
    Discover all Mosaic folders containing required files.
    """
    jobs = []
    for visit_dir in sorted(root.glob("visit_*")):
        if not visit_dir.is_dir():
            continue
        for mosaic_dir in sorted(visit_dir.glob("Mosaic*")):
            if not mosaic_dir.is_dir():
                continue

            flim = mosaic_dir / FNAME_FLIM
            mask = mosaic_dir / FNAME_MASK
            csvc = mosaic_dir / FNAME_CSV_CAL

            if flim.exists() and mask.exists() and csvc.exists():
                jobs.append({
                    "visit": visit_dir.name,
                    "mosaic": mosaic_dir.name,
                    "dir": mosaic_dir,
                    "flim": flim,
                    "mask": mask,
                    "csv_cal": csvc,
                })
    return jobs


# -------------------------
# Step A: GLOBAL reference from calibrated CSVs
# -------------------------
def compute_global_ref_from_calibrated_csvs(jobs: list[dict], out_csv: Path) -> dict:
    rows = []
    for j in jobs:
        df = pd.read_csv(j["csv_cal"])
        class_col = find_class_col(df)
        cls = df[class_col].astype(str).str.strip().str.lower()
        sel = cls == ELASTIN_NAME
        if sel.sum() == 0:
            raise ValueError(f"No elastin rows in {j['csv_cal']}")

        ph, r = get_phase_mod_from_calibrated_csv(df)
        ph = wrap_phase(ph[sel].astype(float))
        r = r[sel].astype(float)

        valid = np.isfinite(ph) & np.isfinite(r)
        ph = ph[valid]
        r = r[valid]

        mu_ph = circular_mean(ph)
        sig_ph = circular_std(ph)
        mu_r = float(np.nanmean(r))
        sig_r = float(np.nanstd(r))

        rows.append({
            "visit": j["visit"],
            "mosaic": j["mosaic"],
            "N": int(ph.size),
            "mu_phase_rad": mu_ph,
            "sigma_phase_rad": sig_ph,
            "mu_r": mu_r,
            "sigma_r": sig_r,
            "csv_path": str(j["csv_cal"]),
        })

        print(
            f"[CAL REF] {j['visit']}|{j['mosaic']}: "
            f"N={ph.size:,} mu_phi={mu_ph:+.6f} sig_phi={sig_ph:.6f} "
            f"mu_r={mu_r:.6f} sig_r={sig_r:.6f}"
        )

    dfv = pd.DataFrame(rows)
    w = dfv["N"].to_numpy(dtype=float)

    mu_phi_bar = float(np.arctan2(np.sum(w * np.sin(dfv["mu_phase_rad"])),
                                  np.sum(w * np.cos(dfv["mu_phase_rad"]))))

    sig_phi_bar = float(np.average(dfv["sigma_phase_rad"], weights=w))
    mu_r_bar = float(np.average(dfv["mu_r"], weights=w))
    sig_r_bar = float(np.average(dfv["sigma_r"], weights=w))

    df_global = pd.DataFrame([{
        "visit": "GLOBAL_REF",
        "mosaic": "",
        "N": int(dfv["N"].sum()),
        "mu_phase_rad": mu_phi_bar,
        "sigma_phase_rad": sig_phi_bar,
        "mu_r": mu_r_bar,
        "sigma_r": sig_r_bar,
        "csv_path": "",
    }])

    out = pd.concat([dfv, df_global], ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    print(f"\n✅ Saved calibrated GLOBAL_REF CSV:\n  {out_csv}")
    print(f"GLOBAL_REF: mu_phi={mu_phi_bar:+.6f} sig_phi={sig_phi_bar:.6f} mu_r={mu_r_bar:.6f} sig_r={sig_r_bar:.6f}")

    return {
        "mu_phi_bar": mu_phi_bar,
        "sig_phi_bar": sig_phi_bar,
        "mu_r_bar": mu_r_bar,
        "sig_r_bar": sig_r_bar,
    }


# -------------------------
# QC plotting with PhasorPlot (universal semicircle)
# -------------------------
def save_elastin_before_after_phasorplot(
    g_raw: np.ndarray, s_raw: np.ndarray,
    g_aut: np.ndarray, s_aut: np.ndarray,
    out_png: Path,
    title: str,
):
    """
    Save a 1x2 PhasorPlot figure (before/after) with universal semicircle.
    """
    # Subsample for plotting
    rng = np.random.default_rng(RNG_SEED)

    def _sub(gv, sv, maxn):
        n = gv.size
        if maxn is None or n <= maxn:
            return gv, sv
        idx = rng.choice(n, size=maxn, replace=False)
        return gv[idx], sv[idx]

    g_raw_p, s_raw_p = _sub(g_raw, s_raw, MAX_POINTS_FOR_QC_SCATTER)
    g_aut_p, s_aut_p = _sub(g_aut, s_aut, MAX_POINTS_FOR_QC_SCATTER)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

    # Before
    pp1 = PhasorPlot(ax=axes[0], frequency=FREQUENCY_MHZ, title="Elastin (UNCAL) — before")
    pp1.hist2d(g_raw_p, s_raw_p)
    pp1.semicircle()

    # After
    pp2 = PhasorPlot(ax=axes[1], frequency=FREQUENCY_MHZ, title="Elastin — after autocal")
    pp2.hist2d(g_aut_p, s_aut_p)
    pp2.semicircle()

    fig.suptitle(title)
    fig.savefig(out_png, dpi=EXPORT_DPI)
    plt.close(fig)


def plot_all_elastin_before_after(
    all_elastin_pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    out_png: Path
):
    """
    all_elastin_pairs: list of (g_raw, s_raw, g_aut, s_aut) arrays
    """
    if not all_elastin_pairs:
        return

    g_raw = np.concatenate([x[0] for x in all_elastin_pairs])
    s_raw = np.concatenate([x[1] for x in all_elastin_pairs])
    g_aut = np.concatenate([x[2] for x in all_elastin_pairs])
    s_aut = np.concatenate([x[3] for x in all_elastin_pairs])

    save_elastin_before_after_phasorplot(
        g_raw, s_raw, g_aut, s_aut,
        out_png,
        title="ALL visits | Elastin phasor before vs after autocalibration"
    )


# -------------------------
# Step B: per-visit autocalibration of uncalibrated FLIM
# -------------------------
def autocalibrate_one(job: dict, ref: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load uncalibrated FLIM
    arr, axes = read_tiff_with_axes(job["flim"])
    flim = to_cyx(arr, axes).astype(np.float32)  # (C,Y,X)

    if flim.shape[0] < GREEN_N_BINS:
        raise ValueError(f"{job['visit']}|{job['mosaic']}: FLIM has C={flim.shape[0]} < {GREEN_N_BINS}")

    green = flim[:GREEN_N_BINS, :, :]  # (16,Y,X)

    # Load elastin mask (binary)
    mask = load_mask_2d(job["mask"])
    if mask.shape != green.shape[1:]:
        raise ValueError(f"{job['visit']}|{job['mosaic']}: mask shape {mask.shape} != YX {green.shape[1:]}")

    # Compute phasor (uncalibrated) from GREEN signal
    mean, g, s = phasor_from_signal(
        green, axis=0, harmonic=1, normalize=True, dtype=np.float32
    )
    phi, r = phasor_to_polar(g, s)
    phi = phi.astype(np.float32)
    r = r.astype(np.float32)

    # Elastin selection (mask + intensity)
    sel = mask & (mean > float(INTENSITY_MIN)) & np.isfinite(phi) & np.isfinite(r)
    if sel.sum() == 0:
        raise ValueError(f"{job['visit']}|{job['mosaic']}: no valid elastin pixels in mask after threshold.")

    phi_e = wrap_phase(phi[sel].ravel())
    r_e = r[sel].ravel()

    # Subsample for stable stats (optional)
    phi_e = subsample_1d(phi_e, MAX_PIXELS_FOR_STATS, RNG_SEED)
    r_e = subsample_1d(r_e, MAX_PIXELS_FOR_STATS, RNG_SEED)

    mu_phi_v = circular_mean(phi_e)
    sig_phi_v = max(circular_std(phi_e), 1e-6)
    mu_r_v = float(np.nanmean(r_e))
    sig_r_v = max(float(np.nanstd(r_e)), 1e-6)

    # ---- standardized mapping (phi1, r1)
    dphi = wrap_phase(phi - mu_phi_v)
    phi1 = wrap_phase((dphi / sig_phi_v) * ref["sig_phi_bar"] + ref["mu_phi_bar"])
    r1 = ((r - mu_r_v) / sig_r_v) * ref["sig_r_bar"] + ref["mu_r_bar"]

    # compute final correction from elastin pixels
    phi1_e = phi1[sel].ravel()
    r1_e = r1[sel].ravel()

    # subsample for stable QC/means
    phi1_e = subsample_1d(phi1_e, MAX_PIXELS_FOR_STATS, RNG_SEED)
    r1_e = subsample_1d(r1_e, MAX_PIXELS_FOR_STATS, RNG_SEED)

    mu_phi1_e = circular_mean(wrap_phase(phi1_e))
    mu_r1_e = float(np.nanmean(r1_e))

    delta_phi = float(wrap_phase(ref["mu_phi_bar"] - mu_phi1_e))
    Fr = float(ref["mu_r_bar"] / (mu_r1_e + 1e-12))

    # ---- apply final calibration to ORIGINAL raw (phi,r)
    phi2 = wrap_phase(phi + delta_phi).astype(np.float32)
    r2 = (Fr * r).astype(np.float32)
    if CLIP_MOD_TO_01:
        r2 = np.clip(r2, 0.0, 1.0)

    g2 = (r2 * np.cos(phi2)).astype(np.float32)
    s2 = (r2 * np.sin(phi2)).astype(np.float32)

    tau_p, tau_m = polar_to_apparent_lifetime(
        phi2, r2, frequency=FREQUENCY_MHZ, unit_conversion=UNIT_CONVERSION
    )

    # Save CYX planes
    out_tif = out_dir / "phasor_uncalibrated_autocal_GREEN_CYX.tif"
    out = np.stack([
        mean.astype(np.float32),
        g2, s2,
        phi2,
        r2,
        np.asarray(tau_p, dtype=np.float32),
        np.asarray(tau_m, dtype=np.float32),
    ], axis=0)

    meta = {
        "axes": "CYX",
        "plane_labels": [
            "green_mean_uncal",
            "green_g_autocal",
            "green_s_autocal",
            "green_phase_autocal_rad",
            "green_mod_autocal",
            "green_tau_phase_autocal_ns",
            "green_tau_mod_autocal_ns",
        ],
        "visit": job["visit"],
        "mosaic": job["mosaic"],
        "global_ref_calibrated": ref,
        "per_visit_uncal_elastin": {
            "mu_phi_v": mu_phi_v,
            "sig_phi_v": sig_phi_v,
            "mu_r_v": mu_r_v,
            "sig_r_v": sig_r_v,
            "mu_phi1_elastin": mu_phi1_e,
            "mu_r1_elastin": mu_r1_e,
            "delta_phi": delta_phi,
            "Fr": Fr,
        },
        "notes": {
            "green_bins": GREEN_N_BINS,
            "intensity_min": float(INTENSITY_MIN),
            "clip_mod_to_01": bool(CLIP_MOD_TO_01),
        }
    }
    tiff.imwrite(str(out_tif), out, description=json.dumps(meta))

    # QC elastin phasor before vs after using PhasorPlot (with universal semicircle)
    g_raw = (r * np.cos(phi))[sel].ravel()
    s_raw = (r * np.sin(phi))[sel].ravel()
    g_aut = g2[sel].ravel()
    s_aut = s2[sel].ravel()

    qc_png = out_dir / "elastin_phasor_before_after.png"
    save_elastin_before_after_phasorplot(
        g_raw, s_raw, g_aut, s_aut,
        qc_png,
        title=f"{job['visit']} | {job['mosaic']}"
    )

    print(f"[OK] {job['visit']}|{job['mosaic']} -> {out_tif.name} | QC: {qc_png.name}")

    return {
        "visit": job["visit"],
        "mosaic": job["mosaic"],
        "dir": str(job["dir"]),
        "out_tif": str(out_tif),
        "qc_png": str(qc_png),
        "N_elastin_pixels_used": int(phi_e.size),
        "mu_phi_v_uncal": mu_phi_v,
        "sig_phi_v_uncal": sig_phi_v,
        "mu_r_v_uncal": mu_r_v,
        "sig_r_v_uncal": sig_r_v,
        "mu_phi1_elastin": mu_phi1_e,
        "mu_r1_elastin": mu_r1_e,
        "delta_phi": delta_phi,
        "Fr": Fr,
    }


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    jobs = discover_jobs(ROOT)
    if not jobs:
        raise FileNotFoundError(
            f"No jobs found under {ROOT}.\n"
            f"Expected each Mosaic folder to contain:\n"
            f"  - {FNAME_FLIM}\n  - {FNAME_MASK}\n  - {FNAME_CSV_CAL}"
        )

    print(f"[INFO] Found {len(jobs)} job(s).")

    # Step A: GLOBAL reference from calibrated CSVs
    ref_csv = OUT_ROOT / "elastin_reference_calibrated_GLOBAL.csv"
    ref = compute_global_ref_from_calibrated_csvs(jobs, ref_csv)

    # Step B: per-visit autocalibration (uncalibrated mosaics)
    stats_rows = []
    all_elastin_pairs = []

    for j in jobs:
        out_dir = OUT_ROOT / j["visit"] / j["mosaic"]

        # Run autocalibration
        row = autocalibrate_one(j, ref, out_dir)
        stats_rows.append(row)

        # Collect elastin before/after points for combined QC plot
        out_arr, _ = read_tiff_with_axes(Path(row["out_tif"]))
        out_cyx = to_cyx(out_arr, "CYX")
        g_aut_img = out_cyx[1]
        s_aut_img = out_cyx[2]

        arr, axes = read_tiff_with_axes(j["flim"])
        flim = to_cyx(arr, axes).astype(np.float32)
        green = flim[:GREEN_N_BINS]
        mask = load_mask_2d(j["mask"])

        mean, g, s = phasor_from_signal(green, axis=0, harmonic=1, normalize=True, dtype=np.float32)
        phi, r = phasor_to_polar(g, s)
        sel = mask & (mean > float(INTENSITY_MIN)) & np.isfinite(phi) & np.isfinite(r)

        g_raw = (r * np.cos(phi))[sel].ravel()
        s_raw = (r * np.sin(phi))[sel].ravel()
        g_aut_e = g_aut_img[sel].ravel()
        s_aut_e = s_aut_img[sel].ravel()

        # subsample to keep memory sane
        g_raw = subsample_1d(g_raw, MAX_POINTS_FOR_QC_SCATTER, RNG_SEED)
        s_raw = subsample_1d(s_raw, MAX_POINTS_FOR_QC_SCATTER, RNG_SEED)
        g_aut_e = subsample_1d(g_aut_e, MAX_POINTS_FOR_QC_SCATTER, RNG_SEED)
        s_aut_e = subsample_1d(s_aut_e, MAX_POINTS_FOR_QC_SCATTER, RNG_SEED)

        all_elastin_pairs.append((g_raw, s_raw, g_aut_e, s_aut_e))

    # Save per-visit stats CSV
    stats_df = pd.DataFrame(stats_rows)
    stats_csv = OUT_ROOT / "per_visit_uncal_elastin_stats.csv"
    stats_df.to_csv(stats_csv, index=False)
    print(f"\n✅ Saved per-visit uncal elastin stats:\n  {stats_csv}")

    # Combined QC plot (PhasorPlot with universal semicircle)
    all_png = OUT_ROOT / "ALL_elastin_before_after.png"
    plot_all_elastin_before_after(all_elastin_pairs, all_png)
    print(f"✅ Saved combined QC plot:\n  {all_png}")

    print("\n✅ Batch autocalibration complete.")
    print(f"📁 Output root: {OUT_ROOT}")


if __name__ == "__main__":
    main()