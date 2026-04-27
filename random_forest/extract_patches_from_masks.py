import re
import numpy as np
import tifffile as tiff
from pathlib import Path
from PIL import Image


VISIT_FOLDER = "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449/visit01"

PATCH_SIZE = 128
STRIDE = 128

SAVE_EMPTY_PATCHES = True
EMPTY_PATCH_KEEP_PROB = 0.4
RANDOM_SEED = 0

# descarta patches con demasiado negro/padding/zonas vacías
MAX_BLACK_FRACTION = 0.30
BLACK_THRESHOLD = 5


def load_rgb(path):
    img = tiff.imread(path)
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"RGB inválido: {path}, shape={img.shape}")
    return img.astype(np.uint8)


def load_mask(path):
    mask = tiff.imread(path)
    mask = np.squeeze(mask)
    return (mask > 0).astype(np.uint8)


def get_tile_id_from_mask(mask_path):
    m = re.search(r"(Im_\d+)", mask_path.stem)
    if m is None:
        raise ValueError(f"No pude detectar tile ID en: {mask_path.name}")
    return m.group(1)


def get_rgb_for_mask(mosaic_dir, mask_path):
    tile_id = get_tile_id_from_mask(mask_path)
    rgb_dir = mosaic_dir / "random_forest" / "rgb"

    candidates = [
        rgb_dir / f"{tile_id}_pseudoRGB.tif",
        rgb_dir / f"{tile_id}_pseudoRGB.tiff",
        rgb_dir / f"{tile_id}_pseudoRGB.png",
    ]

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(f"No encontré RGB para {tile_id} en {rgb_dir}")


def is_bad_patch(img_patch):
    """
    Descarta patches artificiales por padding o casi vacíos.
    """
    black_fraction = np.mean(np.all(img_patch < BLACK_THRESHOLD, axis=-1))
    return black_fraction > MAX_BLACK_FRACTION


def extract_patches_from_pair(rgb_path, mask_path, out_img_dir, out_mask_dir, rng, prefix):
    img = load_rgb(rgb_path)
    mask = load_mask(mask_path)

    if img.shape[:2] != mask.shape:
        raise ValueError(
            f"RGB y máscara tienen distinto tamaño:\n"
            f"RGB: {img.shape[:2]}\n"
            f"Mask: {mask.shape}\n"
            f"RGB path: {rgb_path}\n"
            f"Mask path: {mask_path}"
        )

    h, w = mask.shape

    patch_idx = 1
    n_pos = 0
    n_empty = 0
    n_skipped_black = 0

    # Sin padding: solo patches completamente dentro de la imagen real
    for y in range(0, h - PATCH_SIZE + 1, STRIDE):
        for x in range(0, w - PATCH_SIZE + 1, STRIDE):

            img_patch = img[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            mask_patch = mask[y:y + PATCH_SIZE, x:x + PATCH_SIZE]

            if is_bad_patch(img_patch):
                n_skipped_black += 1
                continue

            has_cells = mask_patch.sum() > 0

            if has_cells:
                n_pos += 1
            else:
                if not SAVE_EMPTY_PATCHES:
                    continue
                if rng.random() > EMPTY_PATCH_KEEP_PROB:
                    continue
                n_empty += 1

            patch_name = f"{prefix}_patch{patch_idx:04d}_y{y:04d}_x{x:04d}.png"
            mask_name = f"{prefix}_patch{patch_idx:04d}_y{y:04d}_x{x:04d}_mask.png"

            Image.fromarray(img_patch).save(out_img_dir / patch_name)
            Image.fromarray((mask_patch * 255).astype(np.uint8)).save(out_mask_dir / mask_name)

            patch_idx += 1

    return n_pos, n_empty, n_skipped_black, patch_idx - 1


def process_visit():
    rng = np.random.default_rng(RANDOM_SEED)
    visit_dir = Path(VISIT_FOLDER)

    mosaic_dirs = sorted([
        p for p in visit_dir.iterdir()
        if p.is_dir() and p.name.lower().startswith("mosaic")
    ])

    total_pos = 0
    total_empty = 0
    total_skipped_black = 0
    total_saved = 0

    print(f"Visit folder: {visit_dir}")
    print(f"Mosaicos encontrados: {len(mosaic_dirs)}")

    for mosaic_dir in mosaic_dirs:
        mask_dir = mosaic_dir / "random_forest" / "mask"

        if not mask_dir.exists():
            print(f"\nSaltando {mosaic_dir.name}: no existe mask/")
            continue

        mask_files = sorted(list(mask_dir.glob("*.tif")) + list(mask_dir.glob("*.tiff")))

        if len(mask_files) == 0:
            print(f"\nSaltando {mosaic_dir.name}: no hay máscaras")
            continue

        out_img_dir = mosaic_dir / "random_forest" / "patch"
        out_mask_dir = mosaic_dir / "random_forest" / "patch_mask"

        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_mask_dir.mkdir(parents=True, exist_ok=True)

        # limpiar outputs previos para no mezclar patches viejos con nuevos
        for old_file in out_img_dir.glob("*.png"):
            old_file.unlink()
        for old_file in out_mask_dir.glob("*.png"):
            old_file.unlink()

        print(f"\nProcesando mosaico: {mosaic_dir.name}")
        print(f"Masks encontradas: {len(mask_files)}")

        for mask_path in mask_files:
            try:
                tile_id = get_tile_id_from_mask(mask_path)
                rgb_path = get_rgb_for_mask(mosaic_dir, mask_path)

                prefix = f"{visit_dir.name}_{mosaic_dir.name}_{tile_id}"

                print(f"  Tile: {tile_id}")
                print(f"    RGB : {rgb_path.name}")
                print(f"    Mask: {mask_path.name}")

                n_pos, n_empty, n_skipped_black, n_saved = extract_patches_from_pair(
                    rgb_path=rgb_path,
                    mask_path=mask_path,
                    out_img_dir=out_img_dir,
                    out_mask_dir=out_mask_dir,
                    rng=rng,
                    prefix=prefix,
                )

                total_pos += n_pos
                total_empty += n_empty
                total_skipped_black += n_skipped_black
                total_saved += n_saved

                print(f"    positivos:       {n_pos}")
                print(f"    vacíos:          {n_empty}")
                print(f"    descartados bg:  {n_skipped_black}")
                print(f"    guardados:       {n_saved}")

            except Exception as e:
                print(f"  ERROR en {mask_path.name}: {e}")

    print("\n======================")
    print("RESUMEN FINAL")
    print("======================")
    print(f"Patches positivos:      {total_pos}")
    print(f"Patches vacíos:         {total_empty}")
    print(f"Patches descartados bg: {total_skipped_black}")
    print(f"Total guardados:        {total_saved}")


if __name__ == "__main__":
    process_visit()