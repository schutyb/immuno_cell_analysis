#!/usr/bin/env python3
"""
Merge two CSVs from visit_04 (4a + 4b) into a single CSV,
reindexing structure_id of 4b so IDs continue after 4a,
and add a source column for traceability.

Output column added:
- visit04_source : "4a" or "4b"

Edit paths in CONFIG and run:
  python merge_visit4_csvs_reindex_ids.py
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd


# ============================================================
# CONFIG
# ============================================================
CSV_4A = Path("/Users/schutyb/Documents/balu_lab/immuno_cell_analysis_data/features_table/structure_features_phasor_classified_4a.csv")
CSV_4B = Path("/Users/schutyb/Documents/balu_lab/immuno_cell_analysis_data/features_table/structure_features_phasor_classified_4b.csv")

OUT_CSV = Path("/Users/schutyb/Documents/balu_lab/immuno_cell_analysis_data/features_table/structure_features_phasor_classified_4.csv")

ID_COL = "structure_id"
SOURCE_COL = "visit04_source"
# ============================================================


def main():
    if not CSV_4A.exists():
        raise FileNotFoundError(f"CSV 4A not found: {CSV_4A}")
    if not CSV_4B.exists():
        raise FileNotFoundError(f"CSV 4B not found: {CSV_4B}")

    print(f"[INFO] Reading CSV 4A: {CSV_4A}")
    df_a = pd.read_csv(CSV_4A)

    print(f"[INFO] Reading CSV 4B: {CSV_4B}")
    df_b = pd.read_csv(CSV_4B)

    if ID_COL not in df_a.columns or ID_COL not in df_b.columns:
        raise ValueError(f"Column '{ID_COL}' must exist in both CSVs.")

    # Ensure IDs are numeric
    df_a[ID_COL] = pd.to_numeric(df_a[ID_COL], errors="raise")
    df_b[ID_COL] = pd.to_numeric(df_b[ID_COL], errors="raise")

    # Add source column (at end later)
    df_a = df_a.copy()
    df_b = df_b.copy()
    df_a[SOURCE_COL] = "4a"
    df_b[SOURCE_COL] = "4b"

    max_id_a = int(df_a[ID_COL].max())
    print(f"[INFO] Max structure_id in 4A: {max_id_a}")

    # Shift IDs in 4B
    df_b[ID_COL] = df_b[ID_COL] + max_id_a

    print(
        f"[INFO] Reindexed 4B IDs: "
        f"{df_b[ID_COL].min()} .. {df_b[ID_COL].max()}"
    )

    # Concatenate
    df_merged = pd.concat([df_a, df_b], ignore_index=True)

    # Ensure SOURCE_COL is last
    cols = [c for c in df_merged.columns if c != SOURCE_COL] + [SOURCE_COL]
    df_merged = df_merged[cols]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(OUT_CSV, index=False)

    print("\n✅ Merge complete")
    print(f"  Rows 4A: {len(df_a)}")
    print(f"  Rows 4B: {len(df_b)}")
    print(f"  Total:   {len(df_merged)}")
    print(f"  Output:  {OUT_CSV}")
    print(f"  Added column: '{SOURCE_COL}' (4a / 4b)")


if __name__ == "__main__":
    main()