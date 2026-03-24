#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

from elastin_correction import (
    compute_reference_from_csv,
    correct_phasor_tiff,
)


# ============================================================
# CONFIG
# ============================================================

PATIENT_DIR = Path("/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449")

# CSV con labels/phasor por ROI
CSV_PATH = PATIENT_DIR / "analysis" / "roi_phasor_points_with_gmm_labels.csv"

# tabla de parámetros de corrección
CORRECTION_TABLE_PATH = PATIENT_DIR / "analysis" / "elastin_reference_params.csv"

# columnas reales de tu CSV
VISIT_COL = "visit"
LABEL_COL = "bio_label"
REFERENCE_LABEL = "elastin"

# primer armónico en tu CSV
G_COL = "g_mean"
S_COL = "s_mean"

# canales del TIFF phasor: (avg, g1, s1, g2, s2)
AVG_CHANNEL = 0
G1_CHANNEL = 1
S1_CHANNEL = 2

# corregir solo donde avg > threshold
VALID_MASK_THRESHOLD = 0.0


# ============================================================
# 1) CALCULAR REFERENCIAS DESDE CSV
# ============================================================

correction_table = compute_reference_from_csv(
    csv_path=CSV_PATH,
    visit_col=VISIT_COL,
    label_col=LABEL_COL,
    reference_label=REFERENCE_LABEL,
    g_col=G_COL,
    s_col=S_COL,
    output_csv=CORRECTION_TABLE_PATH,
)

print("\n=== Correction table ===")
print(correction_table)
print(f"\nSaved correction table to:\n{CORRECTION_TABLE_PATH}")


# ============================================================
# 2) RECORRER VISITAS DEL PACIENTE Y CORREGIR phasor.tif
# ============================================================

visit_dirs = sorted(
    [
        p for p in PATIENT_DIR.iterdir()
        if p.is_dir() and p.name.lower().startswith("visit")
    ]
)

if not visit_dirs:
    raise RuntimeError(f"No visit folders found in: {PATIENT_DIR}")

print(f"\nFound {len(visit_dirs)} visit folders\n")

for visit_dir in visit_dirs:
    visit_name = visit_dir.name  # ej. visit01

    # busca cualquier phasor.tif dentro de esa visita
    phasor_paths = sorted(visit_dir.rglob("phasor.tif"))

    if not phasor_paths:
        print(f"[SKIP] {visit_name}: no phasor.tif found")
        continue

    print(f"[INFO] {visit_name}: found {len(phasor_paths)} phasor.tif files")

    for phasor_path in phasor_paths:
        output_path = phasor_path.with_name("phasor_corr.tif")

        try:
            correct_phasor_tiff(
                phasor_tiff_path=phasor_path,
                correction_table=CORRECTION_TABLE_PATH,
                visit=visit_name,
                visit_col=VISIT_COL,
                output_tiff_path=output_path,
                avg_channel=AVG_CHANNEL,
                g1_channel=G1_CHANNEL,
                s1_channel=S1_CHANNEL,
                valid_mask_threshold=VALID_MASK_THRESHOLD,
            )

            print(f"[OK] {visit_name} -> {output_path}")

        except Exception as e:
            print(f"[ERROR] {phasor_path}")
            print(f"        {type(e).__name__}: {e}")