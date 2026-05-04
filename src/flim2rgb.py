import numpy as np
import tifffile as tiff
from pathlib import Path
from PIL import Image


INPUT_FOLDER = "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449/visit01/Mosaic03_4x4_FOV600_z110_32Sp/flim"
OUTPUT_FOLDER = "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449/visit01/Mosaic03_4x4_FOV600_z110_32Sp/random_forest/rgb"


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

    R = R * SCALE_R
    G = G * SCALE_G
    B = B * SCALE_B

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


def save_mosaic_all_formats(mosaic, output_dir):
    output_dir = Path(output_dir)

    tif_path = output_dir / "mosaic_4x4_pseudoRGB.tif"
    png_path = output_dir / "mosaic_4x4_pseudoRGB.png"
    jpg_path = output_dir / "mosaic_4x4_pseudoRGB.jpg"

    tiff.imwrite(tif_path, mosaic, photometric="rgb")

    pil_img = Image.fromarray(mosaic)
    pil_img.save(png_path, dpi=(600, 600))
    pil_img.save(jpg_path, dpi=(600, 600), quality=95, subsampling=0)

    print(f"Mosaico TIFF guardado: {tif_path}")
    print(f"Mosaico PNG  guardado: {png_path}")
    print(f"Mosaico JPG  guardado: {jpg_path}")


def main():
    input_dir = Path(INPUT_FOLDER)
    output_dir = Path(OUTPUT_FOLDER)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff")))

    print(f"Encontrados {len(files)} archivos TIFF")

    raw_R = []
    raw_G = []
    raw_B = []
    valid_files = []
    rgb_raw_tiles = []

    for f in files:
        try:
            print(f"Leyendo: {f.name}")

            stack = tiff.imread(f)
            R, G, B = flim_to_rgb_raw(stack)

            raw_R.append(R)
            raw_G.append(G)
            raw_B.append(B)
            rgb_raw_tiles.append((R, G, B))
            valid_files.append(f)

        except Exception as e:
            print(f"ERROR en {f.name}: {e}")

    if len(raw_R) == 0:
        raise RuntimeError("No se pudo procesar ningún tile. Revisá los errores anteriores.")

    all_R = np.concatenate([r.ravel() for r in raw_R])
    all_G = np.concatenate([g.ravel() for g in raw_G])
    all_B = np.concatenate([b.ravel() for b in raw_B])

    R_lo, R_hi = np.percentile(all_R, (1, 99))
    G_lo, G_hi = np.percentile(all_G, (1, 99))
    B_lo, B_hi = np.percentile(all_B, (1, 99))

    print("\nPercentiles globales después de scaling:")
    print(f"R: {R_lo:.2f} - {R_hi:.2f}")
    print(f"G: {G_lo:.2f} - {G_hi:.2f}")
    print(f"B: {B_lo:.2f} - {B_hi:.2f}")

    rgb_tiles = []

    for f, (R, G, B) in zip(valid_files, rgb_raw_tiles):
        try:
            Rn = normalize_global(R, R_lo, R_hi)
            Gn = normalize_global(G, G_lo, G_hi)
            Bn = normalize_global(B, B_lo, B_hi)

            rgb = np.stack([Rn, Gn, Bn], axis=-1)
            rgb_uint8 = (rgb * 255).astype(np.uint8)

            out_path = output_dir / f"{f.stem}_pseudoRGB.tif"
            tiff.imwrite(out_path, rgb_uint8, photometric="rgb")

            rgb_tiles.append(rgb_uint8)

            print(f"Guardado: {out_path}")

        except Exception as e:
            print(f"ERROR guardando {f.name}: {e}")

    if len(rgb_tiles) == MOSAIC_ROWS * MOSAIC_COLS:
        print("\nReconstruyendo mosaico 4x4 en orden víbora...")

        mosaic = reconstruct_snake_mosaic(
            rgb_tiles,
            rows=MOSAIC_ROWS,
            cols=MOSAIC_COLS
        )

        save_mosaic_all_formats(mosaic, output_dir)

    else:
        print(
            f"\nNo se reconstruyó mosaico porque hay {len(rgb_tiles)} tiles válidos "
            f"y se esperaban {MOSAIC_ROWS * MOSAIC_COLS}."
        )


if __name__ == "__main__":
    main()