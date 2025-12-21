"""
mask_flim_parameters.py
=======================

Goal
----
Join *morphology* features (from instance masks) with *FLIM lifetime-derived* features
(from calibrated phasor maps), per segmented object.

This script:
1) Traverses the patient folder structure:
      PATIENT_DIR/visit_XX/Mosaic.../mask_...
2) Loads:
   - Instance mask: mask_instances_minEqDiam8px.tif
   - Base morphology CSV: mask_instances_features_minEqDiam8px.csv
3) Locates the calibrated phasor TIFF for the same visit/mosaic:
      PHASOR_ROOT/visit_XX/Mosaic.../phasor_calibrated_CYX.tif
4) Extracts H1 phasor maps (g, s, modulation, phase) from the calibrated CYX TIFF.
5) Computes apparent lifetimes (ns) from polar coordinates:
      tau_phase, tau_mod = polar_to_apparent_lifetime(phase, modulation, frequency=80 MHz)
6) Computes per-object MEANS (using fast bincount):
      g_mean, s_mean, modulation_mean, phase_mean,
      tau_phase_mean_ns, tau_mod_mean_ns
7) Appends these lifetime means to the morphology table and saves:
      mask_instances_features_minEqDiam8px_plusLifetime.csv
8) Filters objects one more time using object-level tau_phase_mean_ns:
      keep TAU_PHASE_MIN <= tau_phase_mean_ns <= TAU_PHASE_MAX
   Then:
   - saves a filtered CSV
   - saves a filtered+relabelled instance mask with only kept objects

Why this step exists
--------------------
- mask_analysis.py produces clean per-object *geometry* features and an instance mask.
- calculate_phasor.py produces *phasor-calibrated* FLIM maps for the mosaic.
This script combines both into one table per mosaic for downstream analysis:
clustering, classifying structures, comparing visits, etc.

Inputs expected
---------------
A) From mask_analysis.py (stored INSIDE each mask_* folder):
   - mask_instances_minEqDiam8px.tif
   - mask_instances_features_minEqDiam8px.csv
   NOTE: The CSV must have a "label" column matching the instance mask labels.

B) From calculate_phasor.py (stored under PHASOR_ROOT):
   - phasor_calibrated_CYX.tif

Calibrated phasor channel layout (CYX)
--------------------------------------
Channel packing used in calculate_phasor.py:

  C0: mean
  C1: g1  (real component, harmonic 1)
  C2: s1  (imag component, harmonic 1)
  C3: mod1
  C4: phase1  (radians)
  C5: g2
  C6: s2
  C7: mod2
  C8: phase2

This script uses ONLY H1 (g1,s1,mod1,phase1).

Outputs
-------
Saved inside each mask_* folder:

1) Extended features table (ALL objects + lifetime means):
   mask_instances_features_minEqDiam8px_plusLifetime.csv

2) Tau-filtered table (subset):
   mask_instances_features_minEqDiam8px_tau0-12.csv

3) Tau-filtered instance mask (relabelled sequentially 1..K):
   mask_instances_minEqDiam8px_tau0-12.tif

Important notes / gotchas
-------------------------
- This script assumes instance mask and phasor maps have identical (Y,X) size.
- Tau filtering is applied using the *MEAN tau_phase per object*.
- After filtering, the instance mask is relabelled sequentially.
  The filtered CSV labels are updated to match the new labels.
- unit_conversion=1e-3 is correct for MHz -> ns in phasorpy:
    MHz * ns = 1e-3 (because 10^6 * 10^-9 = 10^-3)

How to run
----------
Edit PATIENT_DIR (and verify PHASOR_ROOT) then:
    python mask_flim_parameters.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import tifffile as tiff

from phasorpy.lifetime import polar_to_apparent_lifetime
from skimage.segmentation import relabel_sequential


# ============================================================
# CONFIG (edit these parameters)
# ============================================================
PATIENT_DIR = Path("/Users/schutyb/Documents/balu_lab/data_patient_449")

# Phasors are here (confirmed by you):
PHASOR_ROOT = PATIENT_DIR / "phasor_out" / PATIENT_DIR.name

# FLIM settings
FREQUENCY_MHZ = 80.0

# Inputs produced by mask_analysis.py (stored inside each mask_* folder)
INSTANCE_MASK_NAME = "mask_instances_minEqDiam8px.tif"
BASE_FEATURES_CSV_NAME = "mask_instances_features_minEqDiam8px.csv"

# Calibrated phasor filename
PHASOR_CAL_NAME = "phasor_calibrated_CYX.tif"

# Tau filter (ns) using MEAN tau_phase per object
TAU_PHASE_MIN = 0.0
TAU_PHASE_MAX = 12.0

# Outputs saved inside each mask_* folder
OUT_FEATURES_PLUS = "mask_instances_features_minEqDiam8px_plusLifetime.csv"
OUT_FEATURES_TAU  = "mask_instances_features_minEqDiam8px_tau0-12.csv"
OUT_MASK_TAU      = "mask_instances_minEqDiam8px_tau0-12.tif"


# ============================================================
# Folder traversal
# ============================================================
def list_visits(root: Path) -> list[Path]:
    """Return all visit_* folders under PATIENT_DIR."""
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("visit_")])


def list_mosaics(visit_dir: Path) -> list[Path]:
    """Return all Mosaic* folders inside a visit folder."""
    return sorted([p for p in visit_dir.iterdir() if p.is_dir() and p.name.startswith("Mosaic")])


def find_mask_folder(mosaic_dir: Path) -> Path | None:
    """
    Find the mask_* folder inside a Mosaic folder.

    If multiple exist, choose the newest.
    """
    matches = sorted([p for p in mosaic_dir.iterdir() if p.is_dir() and p.name.startswith("mask_")])
    if not matches:
        return None
    if len(matches) > 1:
        matches = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def find_phasor_calibrated(phasor_root: Path, visit_name: str, mosaic_name: str) -> Path | None:
    """
    Find the calibrated phasor TIFF for the given visit/mosaic.

    Expected:
      {phasor_root}/{visit_name}/{mosaic_name}/phasor_calibrated_CYX.tif

    Fallback:
      recursive search under visit folder if folder structure differs by one level.
    """
    cand = phasor_root / visit_name / mosaic_name / PHASOR_CAL_NAME
    if cand.exists():
        return cand

    visit_dir = phasor_root / visit_name
    if visit_dir.exists():
        hits = sorted(visit_dir.rglob(PHASOR_CAL_NAME))
        hits2 = [h for h in hits if mosaic_name in str(h)]
        if hits2:
            return hits2[0]
        if hits:
            return hits[0]

    return None


# ============================================================
# Load phasor tiff (CYX) and extract H1 maps
# ============================================================
def load_phasor_calibrated_h1(path: Path):
    """
    Load calibrated CYX phasor TIFF and return H1 maps (g1,s1,mod1,phase1).

    Returns
    -------
    g1, s1, mod1, phase1 : float32 arrays (Y,X)
    phase is in radians.
    """
    arr = tiff.imread(str(path))
    arr = np.asarray(arr)
    if arr.ndim != 3 or arr.shape[0] < 5:
        raise ValueError(f"Expected (C,Y,X) with >=5 channels. Got {arr.shape} from {path}")

    g1 = arr[1].astype(np.float32)
    s1 = arr[2].astype(np.float32)
    mod1 = arr[3].astype(np.float32)
    phase1 = arr[4].astype(np.float32)  # radians

    return g1, s1, mod1, phase1


# ============================================================
# Fast per-instance mean via bincount
# ============================================================
def mean_per_label(labels: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    Compute mean(values) for each instance label using bincount.

    Parameters
    ----------
    labels : uint32 array (Y,X)
        0 = background, 1..N = instances
    values : float array (Y,X)

    Returns
    -------
    out : float array length (N+1)
        out[label] = mean(values[labels==label])
        out[0] is unused background (NaN).
    """
    if labels.shape != values.shape:
        raise ValueError(f"Shape mismatch labels {labels.shape} vs values {values.shape}")

    lab = labels.ravel()
    val = values.ravel()

    m = lab > 0
    lab = lab[m].astype(np.int64)
    val = val[m].astype(np.float64)

    nlab = int(lab.max())
    sums = np.bincount(lab, weights=val, minlength=nlab + 1)
    cnts = np.bincount(lab, minlength=nlab + 1).astype(np.float64)

    out = np.full(nlab + 1, np.nan, dtype=np.float64)
    good = cnts > 0
    out[good] = sums[good] / cnts[good]
    return out


def filter_and_relabel_instance_mask(inst: np.ndarray, keep_labels: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    """
    Keep only selected labels in an instance mask, then relabel sequentially.

    Parameters
    ----------
    inst : uint32 (Y,X)
    keep_labels : array-like of int
        original labels to keep

    Returns
    -------
    inst_new : uint32 (Y,X)
        filtered + relabeled sequentially (0 bg, 1..K)
    mapping : dict
        old_label -> new_label
    """
    keep_labels = np.asarray(keep_labels, dtype=np.int32)
    if keep_labels.size == 0:
        return np.zeros_like(inst, dtype=np.uint32), {}

    inst_kept = np.where(np.isin(inst, keep_labels), inst, 0).astype(np.int32)
    inst_new, fw, _ = relabel_sequential(inst_kept)

    mapping: dict[int, int] = {}
    for old in keep_labels:
        new = 0
        try:
            new = int(fw(old)) if callable(fw) else int(fw.get(old, 0))  # type: ignore[attr-defined]
        except Exception:
            try:
                new = int(fw[old])  # type: ignore[index]
            except Exception:
                new = 0
        if new > 0:
            mapping[int(old)] = int(new)

    return inst_new.astype(np.uint32), mapping


# ============================================================
# Main
# ============================================================
def main():
    print(f"[INFO] Patient root (masks/csv live here): {PATIENT_DIR}")
    print(f"[INFO] Phasor root (calibrated live here): {PHASOR_ROOT}")

    if not PHASOR_ROOT.exists():
        raise FileNotFoundError(f"PHASOR_ROOT does not exist: {PHASOR_ROOT}")

    visits = list_visits(PATIENT_DIR)
    if not visits:
        raise FileNotFoundError(f"No visit_* folders in {PATIENT_DIR}")

    for visit_dir in visits:
        print(f"\n[VISIT] {visit_dir.name}")

        mosaics = list_mosaics(visit_dir)
        if not mosaics:
            print("  [WARN] No Mosaic* folders")
            continue

        for mosaic_dir in mosaics:
            print(f"  [MOSAIC] {mosaic_dir.name}")

            mask_dir = find_mask_folder(mosaic_dir)
            if mask_dir is None:
                print("    [WARN] No mask_* folder")
                continue

            inst_path = mask_dir / INSTANCE_MASK_NAME
            csv_path  = mask_dir / BASE_FEATURES_CSV_NAME

            if not inst_path.exists():
                print(f"    [WARN] Missing instance mask: {inst_path}")
                continue
            if not csv_path.exists():
                print(f"    [WARN] Missing base features CSV: {csv_path}")
                continue

            phasor_path = find_phasor_calibrated(PHASOR_ROOT, visit_dir.name, mosaic_dir.name)
            if phasor_path is None or not phasor_path.exists():
                print("    [WARN] Missing phasor_calibrated_CYX.tif for this mosaic")
                continue

            print(f"    phasor_path: {phasor_path}")

            # Load instance mask + base features
            inst = tiff.imread(str(inst_path)).astype(np.uint32)
            df = pd.read_csv(csv_path)

            # Load phasor maps (H1)
            g, s, mod, phase = load_phasor_calibrated_h1(phasor_path)

            # Safety check: dimensions must match mosaic
            if inst.shape != g.shape:
                raise ValueError(
                    f"Shape mismatch:\n"
                    f"  inst: {inst.shape}\n"
                    f"  phasor g: {g.shape}\n"
                    f"Paths:\n  {inst_path}\n  {phasor_path}"
                )

            # Apparent lifetimes (ns) from polar coords
            tau_phase, tau_mod = polar_to_apparent_lifetime(
                phase, mod,
                frequency=FREQUENCY_MHZ,
                unit_conversion=1e-3,  # MHz & ns
            )
            tau_phase = np.asarray(tau_phase, dtype=np.float32)
            tau_mod   = np.asarray(tau_mod, dtype=np.float32)

            # Per-instance means
            g_mean   = mean_per_label(inst, g)
            s_mean   = mean_per_label(inst, s)
            mod_mean = mean_per_label(inst, mod)
            ph_mean  = mean_per_label(inst, phase)
            tp_mean  = mean_per_label(inst, tau_phase)
            tm_mean  = mean_per_label(inst, tau_mod)

            # Attach to df by label
            if "label" not in df.columns:
                raise ValueError(f"CSV missing 'label' column: {csv_path}")

            labels = df["label"].astype(int).values
            if labels.size == 0:
                print("    [WARN] CSV has 0 rows; skipping")
                continue

            if labels.max() > inst.max():
                raise ValueError(
                    f"CSV labels exceed mask max label:\n"
                    f"  csv max label = {labels.max()}\n"
                    f"  mask max label = {int(inst.max())}\n"
                    f"  csv: {csv_path}"
                )

            df["g_mean"] = g_mean[labels]
            df["s_mean"] = s_mean[labels]
            df["modulation_mean"] = mod_mean[labels]
            df["phase_mean_rad"] = ph_mean[labels]
            df["tau_phase_mean_ns"] = tp_mean[labels]
            df["tau_mod_mean_ns"] = tm_mean[labels]

            # Save extended CSV (all objects)
            out_plus = mask_dir / OUT_FEATURES_PLUS
            df.to_csv(out_plus, index=False)

            # Filter by tau_phase range (object-level)
            keep = (
                np.isfinite(df["tau_phase_mean_ns"].values) &
                (df["tau_phase_mean_ns"].values >= TAU_PHASE_MIN) &
                (df["tau_phase_mean_ns"].values <= TAU_PHASE_MAX)
            )
            df_tau = df.loc[keep].copy().reset_index(drop=True)

            # Create filtered+relabelled instance mask
            keep_labels = df_tau["label"].astype(int).values
            inst_tau, mapping = filter_and_relabel_instance_mask(inst, keep_labels)

            # Update df_tau labels to match relabeled mask
            df_tau["label_old"] = df_tau["label"]
            df_tau["label"] = df_tau["label_old"].map(lambda x: mapping.get(int(x), 0)).astype(int)
            df_tau = df_tau[df_tau["label"] > 0].sort_values("label").reset_index(drop=True)
            df_tau = df_tau.drop(columns=["label_old"])

            out_tau = mask_dir / OUT_FEATURES_TAU
            df_tau.to_csv(out_tau, index=False)

            out_mask_tau = mask_dir / OUT_MASK_TAU
            tiff.imwrite(
                str(out_mask_tau),
                inst_tau.astype(np.uint32),
                photometric="minisblack",
                metadata={"axes": "YX"},
            )

            print(f"    saved: {out_plus.name} (all objects + lifetime means)")
            print(f"    saved: {out_tau.name}  (tau_phase in [{TAU_PHASE_MIN}, {TAU_PHASE_MAX}] ns) | kept: {len(df_tau)}")
            print(f"    saved: {out_mask_tau.name} | instances kept: {int(inst_tau.max())}")


if __name__ == "__main__":
    main()