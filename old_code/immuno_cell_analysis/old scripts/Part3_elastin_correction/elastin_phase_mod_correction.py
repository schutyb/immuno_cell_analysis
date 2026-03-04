from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# CONFIG  ✅ EDITÁ SOLO ESTO
# ============================================================
INPUT_ROOT = Path("/Users/schutyb/Documents/balu_lab/data_patient_449/phasor_cluster_out")

OUT_ROOT = INPUT_ROOT / "part3_elastin_phase_mod_correction_out"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

COL_G = "g_mean"
COL_S = "s_mean"
COL_PHASE = "phase_mean_rad"        # rad
COL_MOD = "modulation_mean"         # unitless

CLASS_COL = "phasor_class_name"
ELASTIN_NAME = "elastin"

INPUT_PATTERN = "*_phasor_classified.csv"

EPS_SIGMA = 1e-12

COL_G_CORR = f"{COL_G}_corr"
COL_S_CORR = f"{COL_S}_corr"

COL_PHASE_FINAL = f"{COL_PHASE}_final"
COL_MOD_FINAL = f"{COL_MOD}_final"

COL_G_FINAL = f"{COL_G}_final"
COL_S_FINAL = f"{COL_S}_final"


# ============================================================
# Utils
# ============================================================
def safe_numeric(series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def finite_mask(*arrays) -> np.ndarray:
    m = np.ones_like(arrays[0], dtype=bool)
    for a in arrays:
        m &= np.isfinite(a)
    return m


def compute_mu_sigma(x: np.ndarray):
    x = x[np.isfinite(x)]
    if x.size < 5:
        return np.nan, np.nan, int(x.size)
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1)) if x.size > 1 else np.nan
    return mu, sigma, int(x.size)


def apply_distribution_correction(x: np.ndarray, mu_v: float, sig_v: float, mu_g: float, sig_g: float):
    if (not np.isfinite(mu_v)) or (not np.isfinite(sig_v)) or (sig_v < EPS_SIGMA):
        return np.full_like(x, np.nan, dtype=float)
    if (not np.isfinite(mu_g)) or (not np.isfinite(sig_g)) or (sig_g < EPS_SIGMA):
        return np.full_like(x, np.nan, dtype=float)
    return ((x - mu_v) / sig_v) * sig_g + mu_g


def circular_mean(phases: np.ndarray):
    phases = phases[np.isfinite(phases)]
    if phases.size == 0:
        return np.nan
    return float(np.arctan2(np.mean(np.sin(phases)), np.mean(np.cos(phases))))


def compute_phase_mod_params_from_arrays(phase_orig: np.ndarray, mod_orig: np.ndarray,
                                        phase_corr: np.ndarray, mod_corr: np.ndarray):
    mean_ph_o = circular_mean(phase_orig)
    mean_ph_c = circular_mean(phase_corr)

    mean_mo = float(np.mean(mod_orig[np.isfinite(mod_orig)])) if np.any(np.isfinite(mod_orig)) else np.nan
    mean_mc = float(np.mean(mod_corr[np.isfinite(mod_corr)])) if np.any(np.isfinite(mod_corr)) else np.nan

    if not np.isfinite(mean_ph_o) or not np.isfinite(mean_ph_c):
        delta_phase = np.nan
    else:
        diff = float(mean_ph_c - mean_ph_o)
        delta_phase = float(np.arctan2(np.sin(diff), np.cos(diff)))

    if (not np.isfinite(mean_mo)) or (not np.isfinite(mean_mc)) or (mean_mo <= 0):
        mod_factor = np.nan
    else:
        mod_factor = float(mean_mc / mean_mo)

    return {
        "mean_phase_orig": mean_ph_o,
        "mean_phase_corr": mean_ph_c,
        "delta_phase": delta_phase,
        "mean_mod_orig": mean_mo,
        "mean_mod_corr": mean_mc,
        "mod_factor": mod_factor,
    }


# ============================================================
# IO helpers
# ============================================================
def find_input_csvs(input_root: Path) -> list[Path]:
    return sorted(input_root.rglob(INPUT_PATTERN))


def out_dir_for_csv(csv_path: Path, input_root: Path, out_root: Path) -> Path:
    rel_parent = csv_path.parent.relative_to(input_root)
    d = out_root / rel_parent
    d.mkdir(parents=True, exist_ok=True)
    return d


# ============================================================
# Pipeline steps
# ============================================================
def compute_elastin_params(csv_paths: list[Path]) -> tuple[pd.DataFrame, dict]:
    rows = []
    for p in csv_paths:
        df = pd.read_csv(p)

        needed = [CLASS_COL, COL_G, COL_S]
        if any(c not in df.columns for c in needed):
            print(f"[WARN] {p.name}: missing {needed}, skipping")
            continue

        cls = df[CLASS_COL].astype(str).str.lower()
        elast = df[cls == ELASTIN_NAME]

        Ge = safe_numeric(elast[COL_G])
        Se = safe_numeric(elast[COL_S])

        muG, sigG, nG = compute_mu_sigma(Ge)
        muS, sigS, nS = compute_mu_sigma(Se)

        rows.append({
            "file": p.name,
            "path": str(p),
            "n_elastin": int(min(nG, nS)),
            "mu_G_elastin": muG,
            "sigma_G_elastin": sigG,
            "mu_S_elastin": muS,
            "sigma_S_elastin": sigS,
        })

    params_df = pd.DataFrame(rows)

    valid = params_df[
        np.isfinite(params_df["mu_G_elastin"]) &
        np.isfinite(params_df["sigma_G_elastin"]) &
        (params_df["sigma_G_elastin"] > EPS_SIGMA) &
        np.isfinite(params_df["mu_S_elastin"]) &
        np.isfinite(params_df["sigma_S_elastin"]) &
        (params_df["sigma_S_elastin"] > EPS_SIGMA)
    ].copy()

    if valid.empty:
        raise RuntimeError("No valid elastin mu/sigma found (missing elastin or sigma ~ 0).")

    global_params = {
        "muG_avg": float(valid["mu_G_elastin"].mean()),
        "sigG_avg": float(valid["sigma_G_elastin"].mean()),
        "muS_avg": float(valid["mu_S_elastin"].mean()),
        "sigS_avg": float(valid["sigma_S_elastin"].mean()),
    }

    for k, v in global_params.items():
        params_df[k] = v

    return params_df, global_params


def apply_distribution_correction_per_csv(csv_path: Path, row_params: dict, global_params: dict, out_dir: Path) -> Path | None:
    df = pd.read_csv(csv_path)

    needed = [CLASS_COL, COL_G, COL_S, COL_PHASE, COL_MOD]
    if any(c not in df.columns for c in needed):
        print(f"[WARN] {csv_path.name}: missing {needed}, skipping")
        return None

    G = safe_numeric(df[COL_G])
    S = safe_numeric(df[COL_S])
    phase = safe_numeric(df[COL_PHASE])
    mod = safe_numeric(df[COL_MOD])

    valid = finite_mask(G, S, phase, mod)

    muG_v = float(row_params["mu_G_elastin"])
    sigG_v = float(row_params["sigma_G_elastin"])
    muS_v = float(row_params["mu_S_elastin"])
    sigS_v = float(row_params["sigma_S_elastin"])

    Gcorr = np.full_like(G, np.nan, dtype=float)
    Scorr = np.full_like(S, np.nan, dtype=float)

    Gcorr[valid] = apply_distribution_correction(G[valid], muG_v, sigG_v, global_params["muG_avg"], global_params["sigG_avg"])
    Scorr[valid] = apply_distribution_correction(S[valid], muS_v, sigS_v, global_params["muS_avg"], global_params["sigS_avg"])

    phase_corr = np.full_like(phase, np.nan, dtype=float)
    mod_corr = np.full_like(mod, np.nan, dtype=float)

    phase_corr[valid] = np.arctan2(Scorr[valid], Gcorr[valid])
    mod_corr[valid] = np.sqrt(Gcorr[valid] ** 2 + Scorr[valid] ** 2)

    df[COL_G_CORR] = Gcorr
    df[COL_S_CORR] = Scorr
    df[f"{COL_PHASE}_corr"] = phase_corr
    df[f"{COL_MOD}_corr"] = mod_corr

    out_path = out_dir / csv_path.name.replace("_phasor_classified.csv", "_phasor_classified_corrected.csv")
    df.to_csv(out_path, index=False)
    return out_path


def final_phase_mod_calibration(corrected_csv_path: Path, out_dir: Path) -> tuple[Path | None, dict | None]:
    df = pd.read_csv(corrected_csv_path)

    needed = [
        CLASS_COL, COL_PHASE, COL_MOD,
        f"{COL_PHASE}_corr", f"{COL_MOD}_corr",
        COL_G, COL_S
    ]
    if any(c not in df.columns for c in needed):
        print(f"[WARN] {corrected_csv_path.name}: missing {needed}, skipping")
        return None, None

    cls = df[CLASS_COL].astype(str).str.lower().to_numpy()

    phase_o = safe_numeric(df[COL_PHASE])
    mod_o   = safe_numeric(df[COL_MOD])

    phase_c = safe_numeric(df[f"{COL_PHASE}_corr"])
    mod_c   = safe_numeric(df[f"{COL_MOD}_corr"])

    G = safe_numeric(df[COL_G])
    S = safe_numeric(df[COL_S])

    use = finite_mask(phase_o, mod_o, phase_c, mod_c) & (cls == ELASTIN_NAME)
    n_use = int(use.sum())
    if n_use < 30:
        print(f"[WARN] {corrected_csv_path.name}: too few elastin points for calibration (N={n_use}), skipping")
        return None, None

    params = compute_phase_mod_params_from_arrays(
        phase_orig=phase_o[use],
        mod_orig=mod_o[use],
        phase_corr=phase_c[use],
        mod_corr=mod_c[use],
    )

    dph = params["delta_phase"]
    mfac = params["mod_factor"]

    if not np.isfinite(dph) or not np.isfinite(mfac):
        print(f"[WARN] {corrected_csv_path.name}: invalid delta_phase/mod_factor, skipping")
        return None, None

    valid_all = finite_mask(phase_o, mod_o)
    phase_f = np.full_like(phase_o, np.nan, dtype=float)
    mod_f   = np.full_like(mod_o, np.nan, dtype=float)

    phase_f[valid_all] = phase_o[valid_all] + dph
    mod_f[valid_all] = mod_o[valid_all] * mfac

    Gf = np.full_like(G, np.nan, dtype=float)
    Sf = np.full_like(S, np.nan, dtype=float)
    Gf[valid_all] = mod_f[valid_all] * np.cos(phase_f[valid_all])
    Sf[valid_all] = mod_f[valid_all] * np.sin(phase_f[valid_all])

    df[COL_PHASE_FINAL] = phase_f
    df[COL_MOD_FINAL] = mod_f
    df[COL_G_FINAL] = Gf
    df[COL_S_FINAL] = Sf

    out_final = out_dir / corrected_csv_path.name.replace(
        "_phasor_classified_corrected.csv",
        "_phasor_classified_corrected_calibrated.csv"
    )
    df.to_csv(out_final, index=False)

    params_out = {
        "file": corrected_csv_path.name,
        "path": str(corrected_csv_path),
        "n_elastin_used": n_use,
        "n_used": n_use,  # ✅ compat
        **params,
    }
    return out_final, params_out


# ============================================================
# Main
# ============================================================
def main():
    print(f"[INFO] INPUT_ROOT: {INPUT_ROOT}")
    print(f"[INFO] OUT_ROOT:   {OUT_ROOT}")
    print(f"[INFO] Searching pattern: {INPUT_PATTERN}")

    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"INPUT_ROOT does not exist: {INPUT_ROOT}")

    csv_paths = find_input_csvs(INPUT_ROOT)
    print(f"[INFO] Found {len(csv_paths)} classified CSV(s).")
    if not csv_paths:
        raise FileNotFoundError(f"No CSVs found under: {INPUT_ROOT} matching '{INPUT_PATTERN}'")

    params_df, global_params = compute_elastin_params(csv_paths)
    params_df.to_csv(OUT_ROOT / "elastin_correction_params.csv", index=False)

    print("\n[INFO] Global elastin averages:")
    print(global_params)

    lookup = {row["path"]: row for row in params_df.to_dict(orient="records")}

    calib_rows = []
    n_corr = 0
    n_final = 0

    for p in csv_paths:
        row = lookup.get(str(p), None)
        if row is None:
            print(f"[WARN] No elastin params for {p.name}, skipping")
            continue

        out_dir = out_dir_for_csv(p, INPUT_ROOT, OUT_ROOT)

        corr_path = apply_distribution_correction_per_csv(p, row, global_params, out_dir)
        if corr_path is None:
            continue
        n_corr += 1

        final_path, calib_params = final_phase_mod_calibration(corr_path, out_dir)
        if final_path is None or calib_params is None:
            continue
        n_final += 1
        calib_rows.append(calib_params)

        print(f"[OK] {p.name}")
        print(f"     saved corrected:  {corr_path.name}")
        print(f"     saved calibrated: {final_path.name}")
        print(f"     elastin Δphase={calib_params['delta_phase']:.6g} rad | Mfac={calib_params['mod_factor']:.6g} | N={calib_params['n_elastin_used']}")

    calib_df = pd.DataFrame(calib_rows)
    calib_df.to_csv(OUT_ROOT / "final_phase_mod_calibration_params.csv", index=False)

    print("\n✅ DONE Part 3")
    print(f"  corrected CSVs:  {n_corr}")
    print(f"  calibrated CSVs: {n_final}")
    print(f"  outputs in:      {OUT_ROOT}")


if __name__ == "__main__":
    main()