# Creates the RGB image from the flim images 
# with the same normalization across all tiles in a mosaic. 
# Also creates a mosaic of the RGB tiles if all tiles are valid.
# The path take the patient folder as input, 
# and processes all visits and mosaics inside it.


from pathlib import Path
import re

import numpy as np
import tifffile as tiff
from PIL import Image


# =========================
# CONFIG
# =========================

PATIENT_FOLDER = Path(
    "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449"
).expanduser()

FLIM_SUBDIR = "flim"
RGB_OUTPUT_SUBDIR = "RGB"

N_GREEN = 16
N_TOTAL = 31

MOSAIC_ROWS = 4
MOSAIC_COLS = 4

SCALE_R = 4 / 3
SCALE_G = 3 / 3
SCALE_B = 2 / 3

PNG_DPI = 600

PERCENTILE_LOW = 1
PERCENTILE_HIGH = 99


# =========================
# HELPERS
# =========================

def natural_key(path):
    return [
        int(t) if t.isdigit() else t.lower()
        for t in re.split(r"(\d+)", str(path.name))
    ]


def ensure_hwc(stack):
    if stack.ndim != 3:
        raise ValueError(f"Shape inválida: {stack.shape}")

    if stack.shape[0] == N_TOTAL:
        return np.moveaxis(stack, 0, -1)

    if stack.shape[-1] == N_TOTAL:
        return stack

    raise ValueError(
        f"Esperaba {N_TOTAL} canales en eje 0 o eje -1, recibí {stack.shape}"
    )


def flim_to_rgb_raw(stack):
    stack = ensure_hwc(stack)

    green = stack[..., :N_GREEN]
    blue = stack[..., N_GREEN:]

    R = green[..., :2].sum(axis=-1).astype(np.float32)
    G = green[..., 2:].sum(axis=-1).astype(np.float32)
    B = blue.sum(axis=-1).astype(np.float32)

    R *= SCALE_R
    G *= SCALE_G
    B *= SCALE_B

    return R, G, B


def normalize_global(channel, lo, hi):
    if hi <= lo:
        return np.zeros_like(channel, dtype=np.float32)

    channel = np.clip(channel, lo, hi)
    channel = (channel - lo) / (hi - lo)

    return channel.astype(np.float32)


def save_png_600dpi(rgb_uint8, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.fromarray(rgb_uint8)
    img.save(output_path, dpi=(PNG_DPI, PNG_DPI), optimize=True)


def reconstruct_snake_mosaic(tile_images, rows=4, cols=4):
    if len(tile_images) != rows * cols:
        raise ValueError(
            f"Esperaba {rows * cols} tiles, recibí {len(tile_images)}"
        )

    h, w, c = tile_images[0].shape
    mosaic = np.zeros((rows * h, cols * w, c), dtype=np.uint8)

    idx = 0

    for r in range(rows):
        row_tiles = tile_images[idx:idx + cols]

        # Snake acquisition:
        # row 0: left -> right
        # row 1: right -> left
        # row 2: left -> right
        # row 3: right -> left
        if r % 2 == 1:
            row_tiles = row_tiles[::-1]

        for col, tile in enumerate(row_tiles):
            y0 = r * h
            y1 = y0 + h
            x0 = col * w
            x1 = x0 + w

            mosaic[y0:y1, x0:x1, :] = tile

        idx += cols

    return mosaic


def get_flim_files(flim_dir):
    files = list(flim_dir.glob("*.tif")) + list(flim_dir.glob("*.tiff"))
    return sorted(files, key=natural_key)


# =========================
# PROCESSING
# =========================

def process_mosaic_folder(mosaic_dir):
    flim_dir = mosaic_dir / FLIM_SUBDIR
    rgb_out_dir = mosaic_dir / RGB_OUTPUT_SUBDIR

    if not flim_dir.exists():
        print(f"  Saltando, no existe flim/: {flim_dir}")
        return

    files = get_flim_files(flim_dir)

    if len(files) == 0:
        print(f"  Saltando, no hay TIFFs en: {flim_dir}")
        return

    rgb_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcesando mosaico: {mosaic_dir.name}")
    print(f"  FLIM dir: {flim_dir}")
    print(f"  Output RGB dir: {rgb_out_dir}")
    print(f"  TIFFs encontrados: {len(files)}")

    raw_R = []
    raw_G = []
    raw_B = []

    valid_files = []
    rgb_raw_tiles = []

    for f in files:
        try:
            stack = tiff.imread(f)
            R, G, B = flim_to_rgb_raw(stack)

            raw_R.append(R)
            raw_G.append(G)
            raw_B.append(B)

            rgb_raw_tiles.append((R, G, B))
            valid_files.append(f)

        except Exception as e:
            print(f"  ERROR leyendo {f.name}: {e}")

    if len(valid_files) == 0:
        print("  No se pudo procesar ningún tile.")
        return

    all_R = np.concatenate([r.ravel() for r in raw_R])
    all_G = np.concatenate([g.ravel() for g in raw_G])
    all_B = np.concatenate([b.ravel() for b in raw_B])

    R_lo, R_hi = np.percentile(all_R, (PERCENTILE_LOW, PERCENTILE_HIGH))
    G_lo, G_hi = np.percentile(all_G, (PERCENTILE_LOW, PERCENTILE_HIGH))
    B_lo, B_hi = np.percentile(all_B, (PERCENTILE_LOW, PERCENTILE_HIGH))

    print(f"  Percentiles R: {R_lo:.2f} - {R_hi:.2f}")
    print(f"  Percentiles G: {G_lo:.2f} - {G_hi:.2f}")
    print(f"  Percentiles B: {B_lo:.2f} - {B_hi:.2f}")

    rgb_tiles = []

    for f, (R, G, B) in zip(valid_files, rgb_raw_tiles):
        try:
            Rn = normalize_global(R, R_lo, R_hi)
            Gn = normalize_global(G, G_lo, G_hi)
            Bn = normalize_global(B, B_lo, B_hi)

            rgb = np.stack([Rn, Gn, Bn], axis=-1)
            rgb_uint8 = (rgb * 255).round().astype(np.uint8)

            out_png = rgb_out_dir / f"{f.stem}_RGB.png"
            save_png_600dpi(rgb_uint8, out_png)

            rgb_tiles.append(rgb_uint8)

        except Exception as e:
            print(f"  ERROR guardando {f.name}: {e}")

    if len(rgb_tiles) == MOSAIC_ROWS * MOSAIC_COLS:
        try:
            mosaic = reconstruct_snake_mosaic(
                rgb_tiles,
                rows=MOSAIC_ROWS,
                cols=MOSAIC_COLS,
            )

            mosaic_png = rgb_out_dir / f"{mosaic_dir.name}_RGB_mosaic.png"
            save_png_600dpi(mosaic, mosaic_png)

            print(f"  Mosaico reconstruido guardado:")
            print(f"    {mosaic_png}")

        except Exception as e:
            print(f"  ERROR reconstruyendo mosaico: {e}")

    else:
        print(
            f"  No se reconstruyó mosaico: hay {len(rgb_tiles)} tiles válidos "
            f"y se esperaban {MOSAIC_ROWS * MOSAIC_COLS}."
        )


def main():
    patient_dir = PATIENT_FOLDER

    visit_dirs = sorted(
        [
            p for p in patient_dir.iterdir()
            if p.is_dir() and p.name.lower().startswith("visit")
        ],
        key=natural_key,
    )

    print(f"Paciente: {patient_dir}")
    print(f"Visitas encontradas: {[v.name for v in visit_dirs]}")

    for visit_dir in visit_dirs:
        print("\n==============================")
        print(f"Procesando visita: {visit_dir.name}")
        print("==============================")

        mosaic_dirs = sorted(
            [
                p for p in visit_dir.iterdir()
                if p.is_dir() and p.name.lower().startswith("mosaic")
            ],
            key=natural_key,
        )

        print(f"Mosaicos encontrados: {len(mosaic_dirs)}")

        for mosaic_dir in mosaic_dirs:
            process_mosaic_folder(mosaic_dir)

    print("\nListo.")


if __name__ == "__main__":
    main()