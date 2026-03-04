#!/usr/bin/env python3
"""
Elastin correction (GREEN only) across visits, producing "clean" corrected CSVs:
- Do NOT keep old GREEN FLIM/phasor columns in the corrected CSVs.
- Keep morphology + IDs + class columns, but replace green metrics with corrected ones.

Autocalibration note:
- The applied (delta_phase, k_mod) is the autocalibration transform derived from elastin.

Run:
  python elastin_correction_green_only_clean_csv.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from phasorpy.lifetime import polar_to_apparent_lifetime


# ============================================================
# CONFIG (EDIT ME)
# ============================================================
INPUT_DIR = Path("/Users/schutyb/Documents/balu_lab/immuno_cell_analysis_data/features_table")
CSV_GLOB = "structure_features_phasor_classified_*.csv"

CLASS_COL = "phasor_class"
ELASTIN_NAME = "elastin"

# GREEN columns (input)
GREEN_G_IN = "green_g_mean"
GREEN_S_IN = "green_s_mean"
GREEN_INT_IN = "green_intensity_mean"  # optional weighting if present

# Output GREEN columns (we will write these names, without "_corr")
GREEN_INT_OUT = "green_intensity_mean"
GREEN_G_OUT   = "green_g_mean"
GREEN_S_OUT   = "green_s_mean"
GREEN_PH_OUT  = "green_phase_mean"
GREEN_MOD_OUT = "green_modulation_mean"
GREEN_TP_OUT  = "green_tau_phase_mean_ns"
GREEN_TM_OUT  = "green_tau_mod_mean_ns"

# Lifetime conversion
FREQUENCY_MHZ = 80.0
UNIT_CONVERSION = 1e-3  # MHz + ns

# Weighting for elastin center
USE_INTENSITY_WEIGHTS = True

# Output
OUT_DIR = None  # None => INPUT_DIR / "elastin_corrected_out"
OUT_SUFFIX = "_elastinAutocal_GREEN.csv"
SUMMARY_NAME = "elastin_correction_summary_GREEN.csv"

# If True, drop BLUE columns entirely from output CSVs
DROP_BLUE_COLUMNS = True
# ============================================================


def safe_float_series(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def circular_mean(angles: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Weighted circular mean (radians)."""
    if weights is None:
        s = np.nanmean(np.sin(angles))
        c = np.nanmean(np.cos(angles))
    else:
        w = np.asarray(weights, dtype=float)
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        s = np.nansum(w * np.sin(angles))
        c = np.nansum(w * np.cos(angles))
    return float(np.arctan2(s, c))


def compute_elastin_center_polar_green(df: pd.DataFrame) -> tuple[float, float, int]:
    """Return (angle_center, mod_center, n_elastin_used) from GREEN elastin points."""
    if CLASS_COL not in df.columns:
        raise ValueError(f"Missing '{CLASS_COL}' column (GMM classification).")
    if GREEN_G_IN not in df.columns or GREEN_S_IN not in df.columns:
        raise ValueError(f"Missing GREEN phasor columns: '{GREEN_G_IN}', '{GREEN_S_IN}'")

    sel = df[CLASS_COL].astype(str).str.lower() == ELASTIN_NAME
    dfe = df.loc[sel].copy()
    if len(dfe) == 0:
        return float("nan"), float("nan"), 0

    G = safe_float_series(dfe[GREEN_G_IN])
    S = safe_float_series(dfe[GREEN_S_IN])
    valid = np.isfinite(G) & np.isfinite(S)
    if valid.sum() == 0:
        return float("nan"), float("nan"), 0

    G = G[valid]
    S = S[valid]
    ang = np.arctan2(S, G)
    mod = np.sqrt(G * G + S * S)

    w = None
    if USE_INTENSITY_WEIGHTS and (GREEN_INT_IN in dfe.columns):
        w_raw = safe_float_series(dfe.loc[dfe.index[valid], GREEN_INT_IN])
        w = np.where(np.isfinite(w_raw) & (w_raw > 0), w_raw, 0.0)
        if np.nansum(w) <= 0:
            w = None

    ang_c = circular_mean(ang, w)
    if w is None:
        mod_c = float(np.nanmean(mod))
    else:
        mod_c = float(np.nansum(w * mod) / (np.nansum(w) + 1e-12))

    return ang_c, mod_c, int(valid.sum())


def apply_polar_correction(G: np.ndarray, S: np.ndarray, delta: float, k: float):
    """Return corrected (G,S,phase,modulation) after polar correction."""
    ang = np.arctan2(S, G)
    mod = np.sqrt(G * G + S * S)

    ang2 = ang + delta
    mod2 = mod * k

    G2 = mod2 * np.cos(ang2)
    S2 = mod2 * np.sin(ang2)
    return G2, S2, ang2, mod2


def compute_tau_from_polar(phase: np.ndarray, modulation: np.ndarray):
    tau_p, tau_m = polar_to_apparent_lifetime(
        phase,
        modulation,
        frequency=FREQUENCY_MHZ,
        unit_conversion=UNIT_CONVERSION,
    )
    return np.asarray(tau_p, dtype=float), np.asarray(tau_m, dtype=float)


def ensure_out_dir() -> Path:
    out = OUT_DIR if OUT_DIR is not None else (INPUT_DIR / "elastin_corrected_out")
    out.mkdir(parents=True, exist_ok=True)
    return out


def build_clean_output_df(df_in: pd.DataFrame, green_out_cols: dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Create output DataFrame:
    - Remove old GREEN columns (g,s,phase,mod,tau*), then add corrected ones using the standard names.
    - Optionally drop BLUE columns.
    """
    df = df_in.copy()

    # Drop BLUE columns if requested
    if DROP_BLUE_COLUMNS:
        blue_cols = [c for c in df.columns if c.startswith("blue_")]
        if blue_cols:
            df = df.drop(columns=blue_cols)

    # Drop old GREEN metric columns (keep morphology etc.)
    drop_green = [
        "green_g_mean", "green_s_mean",
        "green_phase_mean", "green_modulation_mean",
        "green_tau_phase_mean_ns", "green_tau_mod_mean_ns",
    ]
    # (Intensity we keep as-is; it's not being "corrected" here)
    drop_green = [c for c in drop_green if c in df.columns]
    if drop_green:
        df = df.drop(columns=drop_green)

    # Now append corrected GREEN columns with the standard names
    for colname, values in green_out_cols.items():
        df[colname] = values

    return df


def main():
    csvs = sorted(INPUT_DIR.glob(CSV_GLOB))
    if not csvs:
        raise FileNotFoundError(f"No CSVs found with glob '{CSV_GLOB}' under {INPUT_DIR}")

    out_dir = ensure_out_dir()

    # ---- Pass 1: per-visit elastin centers ----
    centers = []
    for p in csvs:
        df = pd.read_csv(p)
        ang_c, mod_c, n = compute_elastin_center_polar_green(df)
        centers.append({"path": p, "angle": ang_c, "mod": mod_c, "n_elastin": n})

    valid_centers = [c for c in centers if np.isfinite(c["angle"]) and np.isfinite(c["mod"]) and c["n_elastin"] > 0]
    if len(valid_centers) < 2:
        raise ValueError("Too few visits with valid elastin points to compute global target.")

    # Global target (weight by elastin count)
    angles = np.array([c["angle"] for c in valid_centers], dtype=float)
    mods = np.array([c["mod"] for c in valid_centers], dtype=float)
    weights = np.array([c["n_elastin"] for c in valid_centers], dtype=float)

    angle_target = circular_mean(angles, weights)
    mod_target = float(np.average(mods, weights=weights))

    # ---- Apply correction and save ----
    summary_rows = []
    for c in centers:
        p = c["path"]
        if not (np.isfinite(c["angle"]) and np.isfinite(c["mod"]) and c["n_elastin"] > 0):
            print(f"[WARN] Skipping (no valid elastin): {p.name}")
            continue

        df = pd.read_csv(p)

        delta = float(angle_target - c["angle"])
        k = float(mod_target / (c["mod"] + 1e-12))

        G = safe_float_series(df[GREEN_G_IN])
        S = safe_float_series(df[GREEN_S_IN])
        valid = np.isfinite(G) & np.isfinite(S)

        # corrected outputs (nan where invalid)
        Gc = np.full_like(G, np.nan, dtype=float)
        Sc = np.full_like(S, np.nan, dtype=float)
        Phc = np.full_like(G, np.nan, dtype=float)
        Mc = np.full_like(G, np.nan, dtype=float)

        if valid.any():
            G2, S2, ph2, m2 = apply_polar_correction(G[valid], S[valid], delta, k)
            Gc[valid] = G2
            Sc[valid] = S2
            Phc[valid] = ph2
            Mc[valid] = m2

        TPc, TMc = compute_tau_from_polar(Phc, Mc)

        # build "clean" output df (keep morphology, ids, classes; replace green metrics)
        green_out_cols = {
            GREEN_G_OUT: Gc,
            GREEN_S_OUT: Sc,
            GREEN_PH_OUT: Phc,
            GREEN_MOD_OUT: Mc,
            GREEN_TP_OUT: TPc,
            GREEN_TM_OUT: TMc,
        }
        df_clean = build_clean_output_df(df, green_out_cols)

        out_csv = out_dir / f"{p.stem}{OUT_SUFFIX}"
        df_clean.to_csv(out_csv, index=False)

        summary_rows.append({
            "csv": p.name,
            "out_csv": str(out_csv),
            "n_rows": int(len(df)),
            "n_elastin_used": int(c["n_elastin"]),
            "angle_elastin_center_rad": float(c["angle"]),
            "mod_elastin_center": float(c["mod"]),
            "angle_target_rad": float(angle_target),
            "mod_target": float(mod_target),
            "delta_phase_rad": float(delta),
            "k_mod": float(k),
        })

        print(f"[OK] {p.name} -> {out_csv.name} | delta={delta:+.6f} rad | k={k:.6f}")

    summary_path = out_dir / SUMMARY_NAME
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print("\n✅ Elastin correction / autocalibration (GREEN only) done.")
    print(f"   Output dir: {out_dir}")
    print(f"   Summary:    {summary_path}")
    print(f"   Target elastin: angle={angle_target:+.6f} rad | mod={mod_target:.6f}")


if __name__ == "__main__":
    main()