from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tifffile


# ============================================================
# Helpers
# ============================================================

def phasor_to_polar(g: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert Cartesian phasor coordinates (g, s) to polar coordinates
    (modulation, phase).
    """
    g = np.asarray(g, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)

    mod = np.sqrt(g**2 + s**2)
    phi = np.arctan2(s, g)
    return mod, phi


def polar_to_phasor(mod: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert polar phasor coordinates (modulation, phase) back to Cartesian
    phasor coordinates (g, s).
    """
    mod = np.asarray(mod, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)

    g = mod * np.cos(phi)
    s = mod * np.sin(phi)
    return g, s


def circular_mean(angles_rad: np.ndarray) -> float:
    """
    Circular mean of angles in radians.
    """
    angles_rad = np.asarray(angles_rad, dtype=np.float64)
    if angles_rad.size == 0:
        raise ValueError("Cannot compute circular mean of an empty array.")

    return float(np.arctan2(np.mean(np.sin(angles_rad)),
                            np.mean(np.cos(angles_rad))))


# ============================================================
# 1) Compute reference/correction parameters from CSV
# ============================================================

def compute_reference_from_csv(
    csv_path: str | Path,
    *,
    visit_col: str = "visit",
    label_col: str = "bio_label",
    reference_label: str = "elastin",
    g_col: str = "g1_mean",
    s_col: str = "s1_mean",
    output_csv: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Read a CSV of segmented structures and compute visit-level elastin
    correction parameters using only one phasor harmonic (default: first harmonic).

    Parameters
    ----------
    csv_path : str or Path
        Path to input CSV.
    visit_col : str
        Column with visit identifiers.
    label_col : str
        Column with biological labels/classes.
    reference_label : str
        Reference component used to define the longitudinal reference.
        Example: 'elastin'.
    g_col : str
        Column with calibrated g values for the selected harmonic.
        Example: 'g1_mean'.
    s_col : str
        Column with calibrated s values for the selected harmonic.
        Example: 's1_mean'.
    output_csv : str or Path, optional
        If provided, save the correction table to this CSV path.

    Returns
    -------
    correction_table : pd.DataFrame
        Table with one row per visit and columns:
        - visit
        - centroid_g
        - centroid_s
        - sd_g
        - sd_s
        - n_reference
        - mod_visit
        - phi_visit
        - mod_ref
        - phi_ref
        - dphi
        - mod_scale
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    required_cols = {visit_col, label_col, g_col, s_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    # Keep only reference component rows, with valid g/s
    ref_df = df[df[label_col] == reference_label].copy()
    ref_df = ref_df[np.isfinite(ref_df[g_col]) & np.isfinite(ref_df[s_col])].copy()

    if ref_df.empty:
        raise ValueError(
            f"No valid rows found for reference label '{reference_label}' "
            f"using columns '{g_col}' and '{s_col}'."
        )

    # One centroid per visit -> equal weight for each visit
    correction_table = (
        ref_df.groupby(visit_col)
        .agg(
            centroid_g=(g_col, "mean"),
            centroid_s=(s_col, "mean"),
            sd_g=(g_col, "std"),
            sd_s=(s_col, "std"),
            n_reference=(g_col, "size"),
        )
        .reset_index()
    )

    correction_table["sd_g"] = correction_table["sd_g"].fillna(0.0)
    correction_table["sd_s"] = correction_table["sd_s"].fillna(0.0)

    # Convert visit centroids to polar coordinates
    mod_visit, phi_visit = phasor_to_polar(
        correction_table["centroid_g"].to_numpy(),
        correction_table["centroid_s"].to_numpy(),
    )
    correction_table["mod_visit"] = mod_visit
    correction_table["phi_visit"] = phi_visit

    # Global reference: equal-weight average across visits
    mod_ref = float(np.mean(mod_visit))
    phi_ref = circular_mean(phi_visit)

    correction_table["mod_ref"] = mod_ref
    correction_table["phi_ref"] = phi_ref

    # Visit-specific correction parameters
    correction_table["dphi"] = correction_table["phi_ref"] - correction_table["phi_visit"]
    correction_table["mod_scale"] = correction_table["mod_ref"] / correction_table["mod_visit"]

    # Optional save
    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        correction_table.to_csv(output_csv, index=False)

    return correction_table


# ============================================================
# 2) Apply correction to phasor TIFF (avg, g1, s1, g2, s2)
# ============================================================

def correct_phasor_tiff(
    phasor_tiff_path: str | Path,
    correction_table: pd.DataFrame | str | Path,
    *,
    visit,
    visit_col: str = "visit",
    output_tiff_path: Optional[str | Path] = None,
    avg_channel: int = 0,
    g1_channel: int = 1,
    s1_channel: int = 2,
    valid_mask_threshold: Optional[float] = 0.0,
) -> Path:
    """
    Apply visit-specific elastin correction to the first harmonic only (g1, s1)
    in a phasor TIFF stack with channels ordered as (avg, g1, s1, g2, s2).

    Parameters
    ----------
    phasor_tiff_path : str or Path
        Input phasor TIFF path.
    correction_table : pd.DataFrame or str or Path
        Correction table returned by compute_reference_from_csv, or path to it.
    visit :
        Visit identifier used to select row from correction table.
    visit_col : str
        Column name used for visit matching.
    output_tiff_path : str or Path, optional
        Output TIFF path. If None, save as sibling file with suffix '_corr.tif'.
    avg_channel : int
        Channel index of avg.
    g1_channel : int
        Channel index of g1.
    s1_channel : int
        Channel index of s1.
    valid_mask_threshold : float or None
        If not None, apply correction only where avg > threshold.
        If None, apply everywhere finite.

    Returns
    -------
    output_tiff_path : Path
        Saved corrected TIFF path.
    """
    phasor_tiff_path = Path(phasor_tiff_path)

    if isinstance(correction_table, (str, Path)):
        correction_table = pd.read_csv(correction_table)
    else:
        correction_table = correction_table.copy()

    required_corr_cols = {visit_col, "dphi", "mod_scale"}
    missing_corr = required_corr_cols - set(correction_table.columns)
    if missing_corr:
        raise ValueError(f"Missing required columns in correction table: {sorted(missing_corr)}")

    row = correction_table.loc[correction_table[visit_col] == visit]
    if row.empty:
        raise KeyError(f"Visit '{visit}' not found in correction table.")
    if len(row) > 1:
        raise ValueError(f"Visit '{visit}' appears multiple times in correction table.")

    dphi = float(row["dphi"].iloc[0])
    mod_scale = float(row["mod_scale"].iloc[0])

    stack = tifffile.imread(phasor_tiff_path)
    stack = np.asarray(stack)

    if stack.ndim < 3:
        raise ValueError(
            f"Expected phasor TIFF with channel axis first, got shape {stack.shape}"
        )

    n_channels = stack.shape[0]
    for idx in (avg_channel, g1_channel, s1_channel):
        if idx < 0 or idx >= n_channels:
            raise IndexError(f"Channel index {idx} out of bounds for stack with {n_channels} channels")

    # Copy so original is preserved
    corr = stack.astype(np.float32, copy=True)

    avg = corr[avg_channel]
    g1 = corr[g1_channel]
    s1 = corr[s1_channel]

    # Build valid mask
    finite_mask = np.isfinite(g1) & np.isfinite(s1)
    if valid_mask_threshold is None:
        valid_mask = finite_mask
    else:
        valid_mask = finite_mask & np.isfinite(avg) & (avg > valid_mask_threshold)

    # Convert to polar
    mod, phi = phasor_to_polar(g1, s1)

    # Apply correction only where valid
    mod_corr = mod.copy()
    phi_corr = phi.copy()

    mod_corr[valid_mask] = mod[valid_mask] * mod_scale
    phi_corr[valid_mask] = phi[valid_mask] + dphi

    g1_corr, s1_corr = polar_to_phasor(mod_corr, phi_corr)

    corr[g1_channel] = g1_corr.astype(np.float32, copy=False)
    corr[s1_channel] = s1_corr.astype(np.float32, copy=False)

    if output_tiff_path is None:
        output_tiff_path = phasor_tiff_path.with_name(
            phasor_tiff_path.stem + "_corr.tif"
        )
    output_tiff_path = Path(output_tiff_path)
    output_tiff_path.parent.mkdir(parents=True, exist_ok=True)

    tifffile.imwrite(output_tiff_path, corr)

    return output_tiff_path