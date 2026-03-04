#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import tifffile as tiff

from phasorpy.phasor import phasor_from_signal, phasor_to_polar

# ============================================================
# CONFIG  (EDIT THIS)
# ============================================================
PATIENT_DIR = Path("/Users/schutyb/Documents/balu_lab/data_patient_449")

# global elastin reference from Part3
REF_PARAMS_JSON = PATIENT_DIR / "phasor_cluster_out/part3_elastin_phase_mod_correction_out/global_elastin_phase_mod_params.json"

# output
OUT_ROOT = PATIENT_DIR / "part4_global_elastin_normalization_out"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# raw tiles
TILE_GLOB = "Im_*.tif"   # inside each Mosaic folder
TILE_ORDER_4x4 = np.array(
    [
        [1, 2, 3, 4],
        [8, 7, 6, 5],
        [9, 10, 11, 12],
        [16, 15, 14, 13],
    ],
    dtype=int,
)

# choose bins
T_START = 0
T_END = 16
FREQUENCY_MHZ = 80.0

# mask + classified csv names (from your screenshot)
INSTANCE_MASK_NAME = "mask_instances_minEqDiam8px_tau0-12.tif"
CLASSIFIED_CSV_PATTERN = "*_phasor_classified_corrected_calibrated.csv"  # we only need label + class

CLASS_COL = "phasor_class_name"
LABEL_COL = "label"
ELASTIN_NAME = "elastin"

EPS = 1e-12


# ============================================================
# Helpers
# ============================================================
def load_global_params(path: Path) -> dict:
    with open(path, "r") as f:
        d = json.load(f)

    # Accept your JSON keys
    if all(k in d for k in ["mu_mod_ref", "sigma_mod_ref", "mu_phase_ref_rad", "sigma_phase_ref_rad"]):
        return {
            "mu_mod_ref": float(d["mu_mod_ref"]),
            "sigma_mod_ref": float(d["sigma_mod_ref"]),
            "mu_phase_ref_rad": float(d["mu_phase_ref_rad"]),
            "sigma_phase_ref_rad": float(d["sigma_phase_ref_rad"]),
        }

    # Backward-compatible aliases if ever needed
    if all(k in d for k in ["mu_mod_avg", "sig_mod_avg", "mu_phase_avg", "sig_phase_avg"]):
        return {
            "mu_mod_ref": float(d["mu_mod_avg"]),
            "sigma_mod_ref": float(d["sig_mod_avg"]),
            "mu_phase_ref_rad": float(d["mu_phase_avg"]),
            "sigma_phase_ref_rad": float(d["sig_phase_avg"]),
        }

    raise ValueError(
        f"Could not interpret global params JSON keys={list(d.keys())}. "
        "Expected mu_mod_ref/sigma_mod_ref/mu_phase_ref_rad/sigma_phase_ref_rad (your format) "
        "or mu_mod_avg/sig_mod_avg/mu_phase_avg/sig_phase_avg (legacy)."
    )


def load_tile_tyx(path: Path) -> np.ndarray:
    arr = np.asarray(tiff.imread(str(path)))
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D tile, got {arr.shape} at {path}")

    a, b, c = arr.shape
    # detect time axis
    if a <= 256 and b >= 16 and c >= 16:   # (T,Y,X)
        out = arr
    elif c <= 256 and a >= 16 and b >= 16: # (Y,X,T)
        out = np.moveaxis(arr, -1, 0)
    elif b <= 256 and a >= 16 and c >= 16: # (Y,T,X)
        out = np.moveaxis(arr, 1, 0)
    else:
        out = arr
    return out.astype(np.float32)


def assemble_mosaic_4x4_tyx(mosaic_dir: Path) -> np.ndarray:
    tiles = {}
    for i in range(1, 17):
        fp = mosaic_dir / f"Im_{i:05d}.tif"
        if not fp.exists():
            raise FileNotFoundError(f"Missing tile {fp}")
        tiles[i] = load_tile_tyx(fp)

    shapes = {tiles[i].shape for i in range(1, 17)}
    if len(shapes) != 1:
        raise ValueError(f"Tile shapes differ: {shapes}")
    T, ty, tx = next(iter(shapes))

    mos = np.zeros((T, 4 * ty, 4 * tx), dtype=np.float32)
    for r in range(4):
        for c in range(4):
            idx = int(TILE_ORDER_4x4[r, c])
            y0, y1 = r * ty, (r + 1) * ty
            x0, x1 = c * tx, (c + 1) * tx
            mos[:, y0:y1, x0:x1] = tiles[idx]
    return mos


def phasor_maps_from_raw(raw_tyx: np.ndarray):
    T = raw_tyx.shape[0]
    t0 = max(0, int(T_START))
    t1 = min(T, int(T_END))
    if t0 >= t1:
        raise ValueError("Bad T slice")

    decay = raw_tyx[t0:t1].astype(np.float32)
    mean, real, imag = phasor_from_signal(decay, axis=0, harmonic=[1])
    phase, mod = phasor_to_polar(real, imag)  # returns (phase, modulation)

    # real/imag returned with harmonic axis -> (H,Y,X). H=1 here.
    g = real[0].astype(np.float32)
    s = imag[0].astype(np.float32)
    ph = phase[0].astype(np.float32)
    r = mod[0].astype(np.float32)
    m = mean.astype(np.float32)

    return m, g, s, r, ph


def circular_mean(ph: np.ndarray) -> float:
    ph = ph[np.isfinite(ph)]
    if ph.size == 0:
        return np.nan
    return float(np.arctan2(np.mean(np.sin(ph)), np.mean(np.cos(ph))))

def wrap_pi(x: float) -> float:
    return float(np.arctan2(np.sin(x), np.cos(x)))


def compute_elastin_stats_from_mask(mod: np.ndarray, phase: np.ndarray, elastin_pixel_mask: np.ndarray):
    m = np.isfinite(mod) & np.isfinite(phase) & elastin_pixel_mask
    if int(m.sum()) < 50:
        return None

    r = mod[m]
    ph = phase[m]

    mu_r = float(np.mean(r))
    sig_r = float(np.std(r, ddof=1)) if r.size > 1 else np.nan

    # for phase, use circular mean; for sigma use linear std on wrapped residuals
    mu_ph = circular_mean(ph)
    if not np.isfinite(mu_ph):
        return None
    resid = np.arctan2(np.sin(ph - mu_ph), np.cos(ph - mu_ph))
    sig_ph = float(np.std(resid, ddof=1)) if resid.size > 1 else np.nan

    if not (np.isfinite(sig_r) and np.isfinite(sig_ph)) or sig_r < EPS or sig_ph < EPS:
        return None

    return mu_r, sig_r, mu_ph, sig_ph, int(m.sum())


def normalize_to_global(x: np.ndarray, mu0: float, sig0: float, mu_ref: float, sig_ref: float) -> np.ndarray:
    return ((x - mu0) / sig0) * sig_ref + mu_ref


def apply_autocalibration(mod0, ph0, mod1, ph1, elast_mask):
    """
    Compute per-dataset Δφ and Mfac using ELASTIN only:
      Δφ = mean(ph1)-mean(ph0)   (circular)
      M  = mean(mod1)/mean(mod0)
    Then apply to ORIGINAL:
      ph2 = ph0 + Δφ
      r2  = r0 * M
    """
    use = np.isfinite(mod0) & np.isfinite(ph0) & np.isfinite(mod1) & np.isfinite(ph1) & elast_mask
    n = int(use.sum())
    if n < 50:
        return None

    ph0e = ph0[use]; ph1e = ph1[use]
    r0e = mod0[use]; r1e = mod1[use]

    mu_ph0 = circular_mean(ph0e)
    mu_ph1 = circular_mean(ph1e)
    dphi = wrap_pi(mu_ph1 - mu_ph0)

    mu_r0 = float(np.mean(r0e))
    mu_r1 = float(np.mean(r1e))
    if abs(mu_r0) < EPS:
        return None
    mfac = float(mu_r1 / mu_r0)

    ph2 = ph0 + dphi
    r2 = mod0 * mfac
    return dphi, mfac, ph2, r2, n


def save_tiff_cyx(out_path: Path, cyx: np.ndarray):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(str(out_path), cyx.astype(np.float32), photometric="minisblack", metadata={"axes": "CYX"})


def find_visits(patient_dir: Path) -> list[Path]:
    return sorted([p for p in patient_dir.iterdir() if p.is_dir() and p.name.startswith("visit_")])

def find_mosaics(visit_dir: Path) -> list[Path]:
    return sorted([p for p in visit_dir.iterdir() if p.is_dir() and p.name.startswith("Mosaic")])


def build_elastin_pixel_mask(instance_mask_tif: Path, classified_csv: Path) -> np.ndarray:
    """
    instance_mask: label image where pixel value == instance id
    csv: has LABEL_COL and CLASS_COL; elastin rows define which instance ids are elastin
    returns boolean pixel mask for elastin
    """
    inst = np.asarray(tiff.imread(str(instance_mask_tif)))
    inst = np.squeeze(inst)
    if inst.ndim != 2:
        raise ValueError(f"Expected 2D instance mask, got {inst.shape} at {instance_mask_tif}")

    df = pd.read_csv(classified_csv)
    if LABEL_COL not in df.columns or CLASS_COL not in df.columns:
        raise ValueError(f"CSV missing {LABEL_COL} or {CLASS_COL}: {classified_csv}")

    elast_ids = df.loc[df[CLASS_COL].astype(str).str.lower() == ELASTIN_NAME, LABEL_COL]
    elast_ids = pd.to_numeric(elast_ids, errors="coerce").dropna().astype(int).to_numpy()
    if elast_ids.size == 0:
        return np.zeros_like(inst, dtype=bool)

    # fast membership
    elast_set = set(map(int, elast_ids.tolist()))
    mask = np.vectorize(lambda x: int(x) in elast_set)(inst.astype(int))
    return mask.astype(bool)


# ============================================================
# Main
# ============================================================
def main():
    print("[INFO] Part 4 — RAW elastin normalization (phase/mod) -> autocalibrated G/S")
    print(f"[INFO] PATIENT_DIR:     {PATIENT_DIR}")
    print(f"[INFO] OUT_ROOT:        {OUT_ROOT}")
    print(f"[INFO] REF_PARAMS_JSON: {REF_PARAMS_JSON}")

    if not REF_PARAMS_JSON.exists():
        raise FileNotFoundError(f"Missing REF_PARAMS_JSON: {REF_PARAMS_JSON}")

    glob = load_global_params(REF_PARAMS_JSON)
    print("[INFO] Global reference:", glob)

    visits = find_visits(PATIENT_DIR)
    if not visits:
        raise FileNotFoundError("No visit_* folders found.")

    summary_rows = []

    for vdir in visits:
        vname = vdir.name
        for mdir in find_mosaics(vdir):
            mname = mdir.name
            # paths for instance mask + classified csv are in the mosaic folder (per your screenshot)
            inst_mask = mdir / INSTANCE_MASK_NAME
            classified = None
            # classified csv comes from Part3 outputs, which are elsewhere; but you also showed a copy inside mosaic folder.
            # First try inside mosaic folder:
            local = sorted(mdir.glob(CLASSIFIED_CSV_PATTERN))
            if local:
                classified = local[0]
            else:
                # fallback: search in part3 output mirrored structure
                cand = sorted((PATIENT_DIR / "phasor_cluster_out/part3_elastin_phase_mod_correction_out").rglob(f"{mname}/*_phasor_classified_corrected_calibrated.csv"))
                classified = cand[0] if cand else None

            if not inst_mask.exists() or classified is None or not Path(classified).exists():
                print(f"[WARN] {vname}/{mname}: missing instance mask or classified csv -> skip")
                continue

            print(f"\n[RUN] {vname} | {mname}")
            print(f"      mask: {inst_mask}")
            print(f"      csv:  {classified}")

            raw = assemble_mosaic_4x4_tyx(mdir)
            mean, g0, s0, r0, ph0 = phasor_maps_from_raw(raw)

            elast_pix = build_elastin_pixel_mask(inst_mask, Path(classified))

            stats0 = compute_elastin_stats_from_mask(r0, ph0, elast_pix)
            if stats0 is None:
                print("  [WARN] Not enough elastin pixels / invalid stats0 -> skip")
                continue

            mu_r0, sig_r0, mu_ph0, sig_ph0, n_pix = stats0

            # normalize raw to global reference => corrected^1
            r1 = normalize_to_global(r0, mu_r0, sig_r0, glob["mu_mod_ref"], glob["sigma_mod_ref"])
            ph1 = normalize_to_global(ph0, mu_ph0, sig_ph0, glob["mu_phase_ref_rad"], glob["sigma_phase_ref_rad"])

            # autocalibration: compute Δφ and Mfac for THIS dataset, then apply to ORIGINAL => final^2
            cal = apply_autocalibration(r0, ph0, r1, ph1, elast_pix)
            if cal is None:
                print("  [WARN] Could not compute Δφ/Mfac -> skip")
                continue

            dphi, mfac, ph2, r2, n_use = cal

            g2 = r2 * np.cos(ph2)
            s2 = r2 * np.sin(ph2)

            # save CYX: [mean, g0, s0, mod0, ph0, g2, s2, mod2, ph2]
            cyx = np.stack([mean, g0, s0, r0, ph0, g2, s2, r2, ph2], axis=0).astype(np.float32)

            out_dir = OUT_ROOT / vname / mname
            out_dir.mkdir(parents=True, exist_ok=True)
            out_tif = out_dir / "phasor_part4_autocalibrated_CYX.tif"
            save_tiff_cyx(out_tif, cyx)

            summary_rows.append({
                "visit": vname,
                "mosaic": mname,
                "n_elastin_pixels": int(n_pix),
                "n_elastin_used_for_delta": int(n_use),
                "mu_mod_raw": mu_r0,
                "sigma_mod_raw": sig_r0,
                "mu_phase_raw": mu_ph0,
                "sigma_phase_raw": sig_ph0,
                "delta_phase": float(dphi),
                "mod_factor": float(mfac),
                "out_tif": str(out_tif),
            })

            print(f"  [OK] Saved: {out_tif}")
            print(f"       Δphase={dphi:.6g} rad | Mfac={mfac:.6g} | N_elastin={n_use}")

    if summary_rows:
        df = pd.DataFrame(summary_rows).sort_values(["visit", "mosaic"]).reset_index(drop=True)
        out_csv = OUT_ROOT / "part4_summary.csv"
        df.to_csv(out_csv, index=False)
        print(f"\n[SAVED] {out_csv}")
    else:
        print("\n[WARN] No outputs produced. Check paths/names.")

if __name__ == "__main__":
    main()