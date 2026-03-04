"""
Make GLOBAL elastin phase/mod reference JSON from Part 3 outputs.

Reads:
- *_phasor_classified_corrected_calibrated.csv (recursive)
Uses ONLY elastin rows to compute reference distribution on FINAL-calibrated:
  modulation_mean_final, phase_mean_rad_final
(or g_mean_final, s_mean_final -> derive mod/phase)

Outputs:
- global_elastin_phase_mod_params.json

This JSON is used by Part 4 to calibrate new RAW data using elastin ROI statistics.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# CONFIG ✅ EDIT
# ============================================================
INPUT_ROOT = Path(
    "/Users/schutyb/Documents/balu_lab/data_patient_449/phasor_cluster_out/"
    "part3_elastin_phase_mod_correction_out"
)

PATTERN = "*_phasor_classified_corrected_calibrated.csv"

OUT_JSON = INPUT_ROOT / "global_elastin_phase_mod_params.json"

# columns / labels
CLASS_COL = "phasor_class_name"
ELASTIN_NAME = "elastin"

# Prefer these (if present)
MOD_FINAL = "modulation_mean_final"
PH_FINAL  = "phase_mean_rad_final"

# Fallback: compute from g/s final
G_FINAL = "g_mean_final"
S_FINAL = "s_mean_final"

# sanity
MIN_POINTS_PER_VISIT = 2000
EPS = 1e-12


# ============================================================
# Helpers
# ============================================================
def safe_num(a) -> np.ndarray:
    return pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)


def wrap_angle(x: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(x), np.cos(x))


def circular_mean(phi: np.ndarray) -> float:
    phi = phi[np.isfinite(phi)]
    if phi.size == 0:
        return float("nan")
    return float(np.arctan2(np.mean(np.sin(phi)), np.mean(np.cos(phi))))


def circular_std(phi: np.ndarray, mu: float) -> float:
    phi = phi[np.isfinite(phi)]
    if phi.size < 5 or (not np.isfinite(mu)):
        return float("nan")
    d = wrap_angle(phi - mu)
    return float(np.std(d, ddof=1)) if d.size > 1 else float("nan")


def visit_from_path(p: Path) -> str:
    for part in p.parts:
        if part.startswith("visit_"):
            return part
    return "visit_unknown"


# ============================================================
# Main
# ============================================================
def main():
    print(f"[INFO] INPUT_ROOT: {INPUT_ROOT}")
    print(f"[INFO] Searching:  {PATTERN}")

    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"INPUT_ROOT not found: {INPUT_ROOT}")

    csvs = sorted(INPUT_ROOT.rglob(PATTERN))
    print(f"[INFO] Found {len(csvs)} CSV(s).")
    if not csvs:
        raise FileNotFoundError(f"No files found under {INPUT_ROOT} matching {PATTERN}")

    # collect elastin FINAL arrays per visit
    per_visit = {}
    for p in csvs:
        df = pd.read_csv(p)

        if CLASS_COL not in df.columns:
            print(f"[WARN] {p.name}: missing {CLASS_COL} -> skip")
            continue

        cls = df[CLASS_COL].astype(str).str.lower().to_numpy()
        elast = (cls == ELASTIN_NAME)

        if not np.any(elast):
            print(f"[WARN] {p.name}: no elastin rows -> skip")
            continue

        v = visit_from_path(p)

        if (MOD_FINAL in df.columns) and (PH_FINAL in df.columns):
            mod = safe_num(df.loc[elast, MOD_FINAL])
            phi = safe_num(df.loc[elast, PH_FINAL])
        else:
            # fallback compute from g/s final
            missing = [c for c in (G_FINAL, S_FINAL) if c not in df.columns]
            if missing:
                print(f"[WARN] {p.name}: missing {missing} and no mod/phase final -> skip")
                continue
            g = safe_num(df.loc[elast, G_FINAL])
            s = safe_num(df.loc[elast, S_FINAL])
            m = np.isfinite(g) & np.isfinite(s)
            g = g[m]; s = s[m]
            mod = np.sqrt(g**2 + s**2)
            phi = np.arctan2(s, g)

        m2 = np.isfinite(mod) & np.isfinite(phi)
        mod = mod[m2]
        phi = phi[m2]

        if mod.size == 0:
            continue

        per_visit.setdefault(v, {"mod": [], "phi": []})
        per_visit[v]["mod"].append(mod)
        per_visit[v]["phi"].append(phi)

    if not per_visit:
        raise RuntimeError("No elastin FINAL data collected. Check paths/columns/labels.")

    # concatenate per visit, check counts
    visit_stats = []
    all_mod = []
    all_phi = []

    visits = sorted(per_visit.keys())
    print(f"[INFO] Visits discovered: {visits}")

    for v in visits:
        mod_v = np.concatenate(per_visit[v]["mod"]) if per_visit[v]["mod"] else np.array([])
        phi_v = np.concatenate(per_visit[v]["phi"]) if per_visit[v]["phi"] else np.array([])

        n = int(min(mod_v.size, phi_v.size))
        if n < MIN_POINTS_PER_VISIT:
            print(f"[WARN] {v}: elastin points N={n} < {MIN_POINTS_PER_VISIT} (still included, but check)")
        all_mod.append(mod_v)
        all_phi.append(phi_v)

        mu_m = float(np.mean(mod_v)) if mod_v.size else float("nan")
        sd_m = float(np.std(mod_v, ddof=1)) if mod_v.size > 1 else float("nan")
        mu_p = circular_mean(phi_v) if phi_v.size else float("nan")
        sd_p = circular_std(phi_v, mu_p) if phi_v.size else float("nan")

        visit_stats.append({
            "visit": v,
            "n_elastin": n,
            "mu_mod": mu_m,
            "sigma_mod": sd_m,
            "mu_phase_rad": mu_p,
            "sigma_phase_rad": sd_p,
        })

    all_mod = np.concatenate(all_mod)
    all_phi = np.concatenate(all_phi)

    if all_mod.size < 1000:
        raise RuntimeError(f"Too few total elastin points: N={all_mod.size}")

    mu_mod_ref = float(np.mean(all_mod))
    sigma_mod_ref = float(np.std(all_mod, ddof=1))
    mu_phase_ref = float(circular_mean(all_phi))
    sigma_phase_ref = float(circular_std(all_phi, mu_phase_ref))

    if sigma_mod_ref < EPS or sigma_phase_ref < EPS:
        raise RuntimeError("Reference sigma too small — check data consistency.")

    out = {
        "mu_mod_ref": mu_mod_ref,
        "sigma_mod_ref": sigma_mod_ref,
        "mu_phase_ref_rad": mu_phase_ref,
        "sigma_phase_ref_rad": sigma_phase_ref,
        "n_total_elastin": int(all_mod.size),
        "source_root": str(INPUT_ROOT),
        "pattern": PATTERN,
        "per_visit_stats": visit_stats,
    }

    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\n[OK] Saved: {OUT_JSON}")
    print("[GLOBAL REF]")
    print(f"  mu_mod_ref        = {mu_mod_ref:.8f}")
    print(f"  sigma_mod_ref     = {sigma_mod_ref:.8f}")
    print(f"  mu_phase_ref_rad  = {mu_phase_ref:.8f}")
    print(f"  sigma_phase_ref_rad = {sigma_phase_ref:.8f}")
    print(f"  n_total_elastin   = {int(all_mod.size)}")


if __name__ == "__main__":
    main()