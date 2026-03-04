#!/usr/bin/env python3
"""
extract_cells_only_batch_to_one_folder.py

Batch:
- Find all 'structure_features_phasor_classified.csv' under ROOT/visit_*/Mosaic*/
- Filter rows where phasor_class == 'cell'
- Save outputs into ONE folder (OUT_DIR) with visit suffix:
    *_visit01.csv
    *_visit02.csv
    *_visit03.csv
- For visit_04:
    merge BOTH mosaics into a single *_visit04.csv

Outputs (all inside OUT_DIR):
  structure_features_phasor_classified_ONLY_cells_visit01.csv
  structure_features_phasor_classified_ONLY_cells_visit02.csv
  structure_features_phasor_classified_ONLY_cells_visit03.csv
  structure_features_phasor_classified_ONLY_cells_visit04.csv

Run:
  python extract_cells_only_batch_to_one_folder.py
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


# =========================
# CONFIG (EDIT ME)
# =========================
ROOT = Path("/Users/schutyb/Documents/balu_lab/immuno_cell_analysis_data")
CSV_NAME = "structure_features_phasor_classified.csv"

CLASS_COL = "phasor_class"
CELL_LABEL = "cell"

# Save ALL outputs here (single folder)
OUT_DIR = ROOT / "cells_only_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_BASENAME = "structure_features_phasor_classified_ONLY_cells"
# =========================


def discover_csvs(root: Path) -> list[Path]:
    return sorted(root.rglob(CSV_NAME))


def norm(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def parse_visit(csv_path: Path) -> str:
    return next((p for p in csv_path.parts if p.startswith("visit_")), "visit_unknown")


def visit_suffix(visit: str) -> str:
    # visit_01 -> visit01
    if visit.startswith("visit_") and "_" in visit:
        return "visit" + visit.split("_", 1)[1]
    return visit.replace("_", "")


def main():
    csvs = discover_csvs(ROOT)
    if not csvs:
        raise FileNotFoundError(f"No '{CSV_NAME}' found under: {ROOT}")

    print(f"[INFO] Found {len(csvs)} CSV(s)")

    per_visit_cells: dict[str, list[pd.DataFrame]] = {}

    for csv_path in csvs:
        df0 = pd.read_csv(csv_path)

        if CLASS_COL not in df0.columns:
            print(f"[WARN] {csv_path}: missing '{CLASS_COL}'. Skipping.")
            continue

        visit = parse_visit(csv_path)

        # normalize class into helper column
        df0["_class_norm"] = df0[CLASS_COL].map(norm)

        counts = df0["_class_norm"].value_counts(dropna=False)
        print(f"\n=== {visit} | {csv_path.parent.name} ===")
        print("[INFO] Class counts (normalized):")
        print(counts.to_string())

        # filter cells
        df_cell = df0[df0["_class_norm"] == CELL_LABEL].copy()
        print(f"[INFO] Cells kept: {len(df_cell)} / {len(df0)}")

        if df_cell.empty:
            print("[WARN] No cell rows. Skipping.")
            continue

        # drop helper column
        df_cell.drop(columns=["_class_norm"], inplace=True)

        per_visit_cells.setdefault(visit, []).append(df_cell)

    # write outputs (visit_04 mosaics merged automatically)
    for visit, parts in sorted(per_visit_cells.items()):
        df_visit = pd.concat(parts, ignore_index=True)

        out_name = f"{OUT_BASENAME}_{visit_suffix(visit)}.csv"
        out_path = OUT_DIR / out_name
        df_visit.to_csv(out_path, index=False)

        print(f"\n[OK] Saved {out_path}")
        print(f"     rows: {len(df_visit)}")

    print(f"\n✅ Done. Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()