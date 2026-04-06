#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import re
import tifffile

ROOT = Path("/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449")

PIXEL_SIZE = 0.5  # µm

results = []

for visit_dir in sorted(ROOT.glob("visit*")):
    visit = visit_dir.name

    total_cells = 0
    total_area = 0.0
    z_values = []
    mosaic_count = 0

    for mosaic in visit_dir.glob("Mosaic*"):

        props_csv = mosaic / "_new" / "instance_mask_filtered_props.csv"
        mask_path = mosaic / "_new" / "instance_mask_filtered.tif"

        if not props_csv.exists() or not mask_path.exists():
            continue

        mosaic_count += 1

        # ======================
        # count cells
        # ======================
        df = pd.read_csv(props_csv)
        n_cells = len(df)
        total_cells += n_cells

        # ======================
        # extract z
        # ======================
        match = re.search(r"_z(\d+)", mosaic.name)
        if match:
            z_values.append(float(match.group(1)))

        # ======================
        # compute area
        # ======================
        mask = tifffile.imread(mask_path).squeeze()
        ny, nx = mask.shape
        area = (ny * PIXEL_SIZE) * (nx * PIXEL_SIZE)
        total_area += area

    if len(z_values) < 2:
        print(f"[WARN] Not enough z values in {visit}")
        continue

    z_min = min(z_values)
    z_max = max(z_values)
    z_range = z_max - z_min

    volume_um3 = total_area * z_range
    volume_mm3 = volume_um3 / 1e9  # conversión

    density_um3 = total_cells / volume_um3 if volume_um3 > 0 else 0
    density_mm3 = total_cells / volume_mm3 if volume_mm3 > 0 else 0

    results.append({
        "visit": visit,
        "n_mosaics": mosaic_count,
        "total_cells": total_cells,
        "cells_per_mosaic_avg": total_cells / mosaic_count if mosaic_count > 0 else 0,
        "z_min_um": z_min,
        "z_max_um": z_max,
        "z_range_um": z_range,
        "area_um2": total_area,
        "volume_um3": volume_um3,
        "volume_mm3": volume_mm3,
        "density_cells_per_um3": density_um3,
        "density_cells_per_mm3": density_mm3
    })

df_out = pd.DataFrame(results)
out_path = ROOT / "analysis" / "cell_density_by_visit_final.csv"
df_out.to_csv(out_path, index=False)

print("\n=== FINAL TABLE ===")
print(df_out)
print(f"\nSaved to: {out_path}")