import numpy as np
import tifffile as tiff
from pathlib import Path
from PIL import Image


PATIENT_FOLDER = "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449"


N_GREEN = 16
N_TOTAL = 31

MOSAIC_ROWS = 4
MOSAIC_COLS = 4

SCALE_R = 4 / 3
SCALE_G = 3 / 3
SCALE_B = 2 / 3


def ensure_hwc(stack):
    if stack.ndim != 3:
        raise ValueError(f"Shape inválida: {stack.shape}")

    if stack.shape[0] == N_TOTAL:
        return np.moveaxis(stack, 0, -1)

    if stack.shape[-1] == N_TOTAL:
        return stack

    raise ValueError(f"Esperaba {N_TOTAL} canales, recibí {stack.shape}")


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
    return ((channel - lo) / (hi - lo)).astype(np.float32)


def reconstruct_snake_mosaic(tile_images, rows=4, cols=4):
    if len(tile_images) != rows * cols:
        raise ValueError(f"Esperaba {rows * cols} tiles, recibí {len(tile_images)}")

    h, w, c = tile_images[0].shape
    mosaic = np.zeros((rows * h, cols * w, c), dtype=np.uint8)

    idx = 0
    for r in range(rows):
        row_tiles = tile_images[idx:idx + cols]

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


def save_rgb_all_formats(rgb_uint8, output_base):
    output_base = Path(output_base)

    tif_path = output_base.with_suffix(".tif")
    png_path = output_base.with_suffix(".png")
    jpg_path = output_base.with_suffix(".jpg")

    tiff.imwrite(tif_path, rgb_uint8, photometric="rgb")

    pil_img = Image.fromarray(rgb_uint8)
    pil_img.save(png_path, dpi=(600, 600))
    pil_img.save(jpg_path, dpi=(600, 600), quality=95, subsampling=0)


def process_mosaic_folder(mosaic_dir):
    flim_dir = mosaic_dir / "flim"
    rgb_out_dir = mosaic_dir / "random_forest" / "rgb"

    if not flim_dir.exists():
        print(f"  Saltando, no existe flim/: {flim_dir}")
        return

    if not rgb_out_dir.exists():
        print(f"  Saltando, no existe random_forest/rgb/: {rgb_out_dir}")
        return

    files = sorted(list(flim_dir.glob("*.tif")) + list(flim_dir.glob("*.tiff")))

    if len(files) == 0:
        print(f"  Saltando, no hay TIFFs en: {flim_dir}")
        return

    print(f"\nProcesando mosaico: {mosaic_dir.name}")
    print(f"  TIFFs encontrados: {len(files)}")

    raw_R, raw_G, raw_B = [], [], []
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

    if len(raw_R) == 0:
        print("  No se pudo procesar ningún tile.")
        return

    all_R = np.concatenate([r.ravel() for r in raw_R])
    all_G = np.concatenate([g.ravel() for g in raw_G])
    all_B = np.concatenate([b.ravel() for b in raw_B])

    R_lo, R_hi = np.percentile(all_R, (1, 99))
    G_lo, G_hi = np.percentile(all_G, (1, 99))
    B_lo, B_hi = np.percentile(all_B, (1, 99))

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
            rgb_uint8 = (rgb * 255).astype(np.uint8)

            out_base = rgb_out_dir / f"{f.stem}_pseudoRGB"
            save_rgb_all_formats(rgb_uint8, out_base)

            rgb_tiles.append(rgb_uint8)

        except Exception as e:
            print(f"  ERROR guardando {f.name}: {e}")

    if len(rgb_tiles) == MOSAIC_ROWS * MOSAIC_COLS:
        try:
            mosaic = reconstruct_snake_mosaic(
                rgb_tiles,
                rows=MOSAIC_ROWS,
                cols=MOSAIC_COLS
            )

            mosaic_base = rgb_out_dir / "mosaic_4x4_pseudoRGB"
            save_rgb_all_formats(mosaic, mosaic_base)

            print(f"  Mosaico guardado en: {mosaic_base}.tif/.png/.jpg")

        except Exception as e:
            print(f"  ERROR reconstruyendo mosaico: {e}")
    else:
        print(
            f"  No se reconstruyó mosaico: hay {len(rgb_tiles)} tiles válidos "
            f"y se esperaban {MOSAIC_ROWS * MOSAIC_COLS}."
        )


def main():
    patient_dir = Path(PATIENT_FOLDER)

    visit_dirs = sorted([
        p for p in patient_dir.iterdir()
        if p.is_dir() and p.name.lower().startswith("visit")
    ])

    print(f"Paciente: {patient_dir}")
    print(f"Visitas encontradas: {[v.name for v in visit_dirs]}")

    for visit_dir in visit_dirs:
        print(f"\n==============================")
        print(f"Procesando visita: {visit_dir.name}")
        print(f"==============================")

        mosaic_dirs = sorted([
            p for p in visit_dir.iterdir()
            if p.is_dir() and p.name.lower().startswith("mosaic")
        ])

        print(f"Mosaicos encontrados: {len(mosaic_dirs)}")

        for mosaic_dir in mosaic_dirs:
            process_mosaic_folder(mosaic_dir)

    print("\nListo.")


if __name__ == "__main__":
    main()