from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd


# ============================================================
# CONFIG  ✅ EDITÁ SOLO ESTO
# ============================================================
INPUT_ROOT = Path(
    "/Users/schutyb/Documents/balu_lab/data_patient_449/"
    "phasor_cluster_out/part3_elastin_phase_mod_correction_out"
)

REF_PARAMS_JSON = INPUT_ROOT / "global_elastin_phase_mod_params.json"

OUT_DIR = INPUT_ROOT / "per_visit_phase_mod_params"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATTERN = "*_phasor_classified_corrected_calibrated.csv"

CLASS_COL = "phasor_class_name"
ELASTIN_NAME = "elastin"

# ORIGINAL and CORRECTED
G0, S0 = "g_mean", "s_mean"
GC, SC = "g_mean_corr", "s_mean_corr"


# ============================================================
# Helpers
# ============================================================
def safe_numeric(x) -> np.ndarray:
    return pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)

def finite_mask(*arrs) -> np.ndarray:
    m = np.ones_like(arrs[0], dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return m

def wrap_pi(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))

def circular_mean(phases: np.ndarray) -> float:
    phases = phases[np.isfinite(phases)]
    if phases.size == 0:
        return np.nan
    return float(np.arctan2(np.mean(np.sin(phases)), np.mean(np.cos(phases))))

def phasor_to_phase_mod(G: np.ndarray, S: np.ndarray):
    phase = np.arctan2(S, G)
    mod = np.sqrt(G**2 + S**2)
    return phase, mod

def visit_from_path(p: Path) -> str:
    for part in p.parts:
        if part.startswith("visit_"):
            return part
    return "visit_unknown"

def load_global_params(json_path: Path) -> dict:
    """
    Supports multiple JSON schemas.
    Returns dict with:
      mu_phase_ref, sig_phase_ref, mu_mod_ref, sig_mod_ref
    or {} if not recognized.
    """
    if not json_path.exists():
        print(f"[WARN] Global JSON not found: {json_path}")
        return {}

    d = json.loads(json_path.read_text())

    # Schema A (older):
    keysA = {"mu_mod_avg", "sig_mod_avg", "mu_phase_avg", "sig_phase_avg"}
    if keysA.issubset(d.keys()):
        return {
            "mu_mod_ref": float(d["mu_mod_avg"]),
            "sig_mod_ref": float(d["sig_mod_avg"]),
            "mu_phase_ref": float(d["mu_phase_avg"]),
            "sig_phase_ref": float(d["sig_phase_avg"]),
        }

    # Schema B (your current JSON):
    keysB = {"mu_mod_ref", "sigma_mod_ref", "mu_phase_ref_rad", "sigma_phase_ref_rad"}
    if keysB.issubset(d.keys()):
        return {
            "mu_mod_ref": float(d["mu_mod_ref"]),
            "sig_mod_ref": float(d["sigma_mod_ref"]),
            "mu_phase_ref": float(d["mu_phase_ref_rad"]),
            "sig_phase_ref": float(d["sigma_phase_ref_rad"]),
        }

    print(f"[WARN] Global JSON exists but keys not recognized. Keys={list(d.keys())}")
    return {}


# ============================================================
# Main
# ============================================================
def main():
    print("[INFO] Computing Δphase and Mfac per visit (ELASTIN-only)")
    print(f"[INFO] INPUT_ROOT: {INPUT_ROOT}")
    print(f"[INFO] Pattern:    {PATTERN}")

    files = sorted(INPUT_ROOT.rglob(PATTERN))
    if not files:
        raise FileNotFoundError(f"No files found under {INPUT_ROOT} matching {PATTERN}")

    print(f"[INFO] Found {len(files)} file(s)")

    per_visit: dict[str, list[Path]] = {}
    for p in files:
        v = visit_from_path(p)
        per_visit.setdefault(v, []).append(p)

    visits = sorted(per_visit.keys())
    print(f"[INFO] Visits: {visits}")

    glob = load_global_params(REF_PARAMS_JSON)
    if glob:
        print("[INFO] Loaded global reference params:")
        print(f"       mu_phase_ref={glob['mu_phase_ref']:.6g} rad | mu_mod_ref={glob['mu_mod_ref']:.6g}")

    rows = []

    for v in visits:
        dfs = []
        for p in per_visit[v]:
            df = pd.read_csv(p)

            needed = [CLASS_COL, G0, S0, GC, SC]
            miss = [c for c in needed if c not in df.columns]
            if miss:
                print(f"[WARN] {p.name} missing {miss} -> skipping this file")
                continue

            df["_source_csv"] = str(p)
            dfs.append(df)

        if not dfs:
            print(f"[WARN] {v}: no usable CSVs after filtering")
            continue

        dfv = pd.concat(dfs, ignore_index=True)

        cls = dfv[CLASS_COL].astype(str).str.lower().to_numpy()
        elast = (cls == ELASTIN_NAME)

        g0 = safe_numeric(dfv[G0]); s0 = safe_numeric(dfv[S0])
        gc = safe_numeric(dfv[GC]); sc = safe_numeric(dfv[SC])

        # masks
        m_use = finite_mask(g0, s0, gc, sc) & elast
        n_use = int(m_use.sum())
        n0 = int((finite_mask(g0, s0) & elast).sum())
        nc = int((finite_mask(gc, sc) & elast).sum())

        if n_use < 30:
            print(f"[WARN] {v}: too few elastin points for robust params (N={n_use}). Skipping.")
            continue

        # phase/mod from G,S (orig + corr) using same elastin mask
        ph0, mo0 = phasor_to_phase_mod(g0[m_use], s0[m_use])
        phc, moc = phasor_to_phase_mod(gc[m_use], sc[m_use])

        mean_ph0 = circular_mean(ph0)
        mean_phc = circular_mean(phc)
        mean_mo0 = float(np.mean(mo0[np.isfinite(mo0)]))
        mean_moc = float(np.mean(moc[np.isfinite(moc)]))

        # ORIGINAL -> CORRECTED
        delta_phase = wrap_pi(mean_phc - mean_ph0)
        mod_factor = float(mean_moc / mean_mo0) if (np.isfinite(mean_moc) and np.isfinite(mean_mo0) and mean_mo0 > 0) else np.nan

        out = dict(
            visit=v,
            n_elastin_orig=n0,
            n_elastin_corr=nc,
            n_elastin_used=n_use,
            mean_phase_orig=mean_ph0,
            mean_phase_corr=mean_phc,
            delta_phase_orig_to_corr=delta_phase,
            mean_mod_orig=mean_mo0,
            mean_mod_corr=mean_moc,
            mod_factor_orig_to_corr=mod_factor,
        )

        # OPTIONAL: ORIGINAL -> GLOBAL reference
        if glob:
            out["delta_phase_orig_to_global"] = wrap_pi(glob["mu_phase_ref"] - mean_ph0)
            out["mod_factor_orig_to_global"] = float(glob["mu_mod_ref"] / mean_mo0) if (np.isfinite(mean_mo0) and mean_mo0 > 0) else np.nan

        rows.append(out)

        print(f"[OK] {v}: Δphase(orig→corr)={delta_phase:.6g} rad | Mfac(orig→corr)={mod_factor:.6g} | N={n_use}")

    if not rows:
        raise RuntimeError("No per-visit parameters computed. Check labels/columns/paths.")

    df_out = pd.DataFrame(rows).sort_values("visit").reset_index(drop=True)
    out_csv = OUT_DIR / "per_visit_delta_phase_mod_factor_elastin.csv"
    df_out.to_csv(out_csv, index=False)

    print(f"\n[SAVED] {out_csv}")
    print(f"[DONE] Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()