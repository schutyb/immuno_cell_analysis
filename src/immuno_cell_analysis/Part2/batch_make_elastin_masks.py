#!/usr/bin/env python3
"""
Batch build ELASTIN-only masks from:
  - structure_features_phasor_classified*.csv (per-structure table w/ structure_id + class)
  - structure_instance_mask_uint32.tif        (label image: pixel value = structure_id)

Root structure:
  ROOT/visit_01/Mosaic*/...
  ROOT/visit_02/Mosaic*/...
  ROOT/visit_03/Mosaic*/...
  ROOT/visit_04/Mosaic*/... (may contain multiple CSVs/masks)

Behavior:
- For each Mosaic folder, finds ALL matching CSVs:
    structure_features_phasor_classified*.csv
  (So visit_04 with 2 CSVs is handled automatically.)
- For each CSV -> creates:
    elastin_mask_mosaic_<csv_stem>.png
- If multiple CSVs exist -> merges elastin ids across CSVs -> creates:
    elastin_mask_mosaic_merged.png
- Finally writes an alias used by downstream scripts:
    elastin_mask_mosaic.png
  (merged if available, else the single mask)

Edit ROOT and run:
  python batch_make_elastin_masks.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import tifffile as tiff
import imageio.v3 as iio


# ============================================================
# CONFIG (EDIT ME)
# ============================================================
ROOT = Path("/Users/schutyb/Documents/balu_lab/immuno_cell_analysis_data")

CSV_GLOB = "structure_features_phasor_classified*.csv"
INSTANCE_TIF_NAME = "structure_instance_mask_uint32.tif"

# candidates
ID_COL_CANDIDATES = ["structure_id", "label", "id", "instance_id"]
CLASS_COL_CANDIDATES = ["phasor_class", "phasor_class_name"]

ELASTIN_LABEL = "elastin"
# ============================================================


def find_col(df: pd.DataFrame, candidates: list[str], kind: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find {kind} column. Tried {candidates}. Columns: {list(df.columns)}")


def to_int_ids(series: pd.Series) -> np.ndarray:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return s.astype(np.int64).to_numpy()


def load_instance_mask(path: Path) -> np.ndarray:
    inst = tiff.imread(str(path))
    if inst.ndim == 3 and inst.shape[0] == 1:
        inst = inst[0]
    if inst.ndim != 2:
        raise ValueError(f"{path}: expected 2D instance mask, got shape {inst.shape}")
    return inst.astype(np.int64)


def save_png_mask(path: Path, mask_bool: np.ndarray):
    out = (mask_bool.astype(np.uint8) * 255)
    iio.imwrite(path, out)


def iter_mosaic_dirs(root: Path) -> list[Path]:
    mosaics = []
    for vdir in sorted(root.glob("visit_*")):
        if not vdir.is_dir():
            continue
        for mdir in sorted(vdir.glob("Mosaic*")):
            if not mdir.is_dir():
                continue
            if (mdir / INSTANCE_TIF_NAME).exists():
                mosaics.append(mdir)
    return mosaics


def main():
    mosaic_dirs = iter_mosaic_dirs(ROOT)
    if not mosaic_dirs:
        raise FileNotFoundError(f"No Mosaic* folders with {INSTANCE_TIF_NAME} found under {ROOT}")

    print(f"[INFO] Found {len(mosaic_dirs)} mosaic folder(s) under {ROOT}")

    n_saved = 0
    for mdir in mosaic_dirs:
        csv_paths = sorted(mdir.glob(CSV_GLOB))
        if not csv_paths:
            print(f"[WARN] {mdir}: no '{CSV_GLOB}' found. Skipping.")
            continue

        inst_path = mdir / INSTANCE_TIF_NAME
        inst = load_instance_mask(inst_path)

        elastin_masks = []
        elastin_ids_all = []

        print(f"\n[MOSAIC] {mdir}")
        print(f"  instance mask: {inst_path.name}")
        print(f"  CSVs: {len(csv_paths)}")

        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)

            id_col = find_col(df, ID_COL_CANDIDATES, "ID")
            class_col = find_col(df, CLASS_COL_CANDIDATES, "class")

            classes = df[class_col].astype(str).str.strip().str.lower()
            sel = classes == ELASTIN_LABEL

            if sel.sum() == 0:
                print(f"  [WARN] {csv_path.name}: no elastin rows. Skipping this CSV.")
                continue

            elastin_ids = np.unique(to_int_ids(df.loc[sel, id_col]))
            elastin_ids_all.append(elastin_ids)

            mask_e = np.isin(inst, elastin_ids)
            elastin_masks.append(mask_e)

            out_single = mdir / f"elastin_mask_mosaic_{csv_path.stem}.png"
            save_png_mask(out_single, mask_e)

            print(f"  [SAVED] {out_single.name} | elastin IDs={elastin_ids.size:,} | pixels={int(mask_e.sum()):,}")
            n_saved += 1

        if not elastin_masks:
            print(f"  [WARN] No elastin masks produced in {mdir}.")
            continue

        # Merge if multiple CSVs yielded masks
        if len(elastin_masks) > 1:
            merged = np.logical_or.reduce(elastin_masks)
            out_merged = mdir / "elastin_mask_mosaic_merged.png"
            save_png_mask(out_merged, merged)
            print(f"  [SAVED] {out_merged.name} | pixels={int(merged.sum()):,}")
            n_saved += 1

            # Alias
            alias = mdir / "elastin_mask_mosaic.png"
            save_png_mask(alias, merged)
            print(f"  [ALIAS] {alias.name} -> merged")
            n_saved += 1
        else:
            # only one
            alias = mdir / "elastin_mask_mosaic.png"
            save_png_mask(alias, elastin_masks[0])
            print(f"  [ALIAS] {alias.name} -> single")
            n_saved += 1

    print(f"\n✅ Done. Wrote {n_saved} PNG(s).")


if __name__ == "__main__":
    main()