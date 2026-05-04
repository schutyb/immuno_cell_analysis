import re
from pathlib import Path

import numpy as np
import tifffile as tiff
from PIL import Image


# =========================
# CONFIG
# =========================

PATIENT_FOLDER = "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449"

PATCH_SIZE = 128
STRIDE = 128

# If True, clears old patches before writing new ones
CLEAR_OLD_PATCHES = True


# =========================
# IO
# =========================

def load_rgb(path):
    path = Path(path)

    if path.suffix.lower() in [".tif", ".tiff"]:
        img = tiff.imread(path)
    else:
        img = np.array(Image.open(path).convert("RGB"))

    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Invalid RGB image: {path}, shape={img.shape}")

    return img.astype(np.uint8)


def load_mask(path):
    mask = tiff.imread(path)
    mask = np.squeeze(mask)
    return (mask > 0).astype(np.uint8)


# =========================
# MATCH RGB TILE TO MASK TILE
# =========================

def get_tile_id_from_mask(mask_path):
    """
    Supports names like:
    - Im_00001_mask.tif
    - immune_cells_mask_Im_00001.tiff
    """
    match = re.search(r"(Im_\d+)", mask_path.stem)

    if match is None:
        raise ValueError(f"Could not detect tile ID from mask name: {mask_path.name}")

    return match.group(1)


def get_rgb_for_mask(random_forest_dir, mask_path):
    tile_id = get_tile_id_from_mask(mask_path)
    rgb_dir = random_forest_dir / "rgb"

    candidates = [
        rgb_dir / f"{tile_id}_pseudoRGB.tif",
        rgb_dir / f"{tile_id}_pseudoRGB.tiff",
        rgb_dir / f"{tile_id}_pseudoRGB.png",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"No RGB image found for {tile_id} in {rgb_dir}")


# =========================
# PATCH EXTRACTION
# =========================

def extract_positive_patches(rgb_path, mask_path, out_img_dir, out_mask_dir, prefix):
    img = load_rgb(rgb_path)
    mask = load_mask(mask_path)

    if img.shape[:2] != mask.shape:
        raise ValueError(
            f"RGB and mask size mismatch:\n"
            f"RGB:  {img.shape[:2]}\n"
            f"Mask: {mask.shape}\n"
            f"RGB path:  {rgb_path}\n"
            f"Mask path: {mask_path}"
        )

    h, w = mask.shape

    patch_idx = 1
    n_positive = 0

    # No padding: only full 128x128 patches inside real image area
    for y in range(0, h - PATCH_SIZE + 1, STRIDE):
        for x in range(0, w - PATCH_SIZE + 1, STRIDE):

            mask_patch = mask[y:y + PATCH_SIZE, x:x + PATCH_SIZE]

            # Save ONLY patches containing annotated cells
            if mask_patch.sum() == 0:
                continue

            img_patch = img[y:y + PATCH_SIZE, x:x + PATCH_SIZE]

            patch_name = f"{prefix}_patch{patch_idx:04d}_y{y:04d}_x{x:04d}.png"
            mask_name = f"{prefix}_patch{patch_idx:04d}_y{y:04d}_x{x:04d}_mask.png"

            Image.fromarray(img_patch).save(out_img_dir / patch_name)
            Image.fromarray((mask_patch * 255).astype(np.uint8)).save(out_mask_dir / mask_name)

            patch_idx += 1
            n_positive += 1

    return n_positive


# =========================
# FOLDER WALK
# =========================

def find_random_forest_dirs(patient_folder):
    patient_folder = Path(patient_folder)

    return sorted([
        p for p in patient_folder.glob("visit*/Mosaic*/random_forest")
        if p.is_dir()
    ])


def clear_old_outputs(out_img_dir, out_mask_dir):
    if not CLEAR_OLD_PATCHES:
        return

    for f in out_img_dir.glob("*.png"):
        f.unlink()

    for f in out_mask_dir.glob("*.png"):
        f.unlink()


def process_random_forest_dir(random_forest_dir):
    mask_dir = random_forest_dir / "mask"
    rgb_dir = random_forest_dir / "rgb"

    if not mask_dir.exists():
        print(f"Skipping: no mask folder → {random_forest_dir}")
        return 0

    if not rgb_dir.exists():
        print(f"Skipping: no rgb folder → {random_forest_dir}")
        return 0

    mask_files = sorted(list(mask_dir.glob("*.tif")) + list(mask_dir.glob("*.tiff")))

    if len(mask_files) == 0:
        print(f"Skipping: no masks found → {mask_dir}")
        return 0

    out_img_dir = random_forest_dir / "patch"
    out_mask_dir = random_forest_dir / "patch_mask"

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    clear_old_outputs(out_img_dir, out_mask_dir)

    mosaic_dir = random_forest_dir.parent
    visit_dir = mosaic_dir.parent

    print(f"\nProcessing:")
    print(f"  Visit:  {visit_dir.name}")
    print(f"  Mosaic: {mosaic_dir.name}")
    print(f"  Masks:  {len(mask_files)}")

    total_positive = 0

    for mask_path in mask_files:
        try:
            tile_id = get_tile_id_from_mask(mask_path)
            rgb_path = get_rgb_for_mask(random_forest_dir, mask_path)

            prefix = f"{visit_dir.name}_{mosaic_dir.name}_{tile_id}"

            n_positive = extract_positive_patches(
                rgb_path=rgb_path,
                mask_path=mask_path,
                out_img_dir=out_img_dir,
                out_mask_dir=out_mask_dir,
                prefix=prefix,
            )

            total_positive += n_positive

            print(f"  {tile_id}: {n_positive} positive patches")

        except Exception as e:
            print(f"  ERROR with {mask_path.name}: {e}")

    print(f"  Total positive patches: {total_positive}")

    return total_positive


def main():
    patient_dir = Path(PATIENT_FOLDER)

    random_forest_dirs = find_random_forest_dirs(patient_dir)

    print(f"Patient folder: {patient_dir}")
    print(f"random_forest folders found: {len(random_forest_dirs)}")

    total_patches = 0

    for rf_dir in random_forest_dirs:
        total_patches += process_random_forest_dir(rf_dir)

    print("\n======================")
    print("FINAL SUMMARY")
    print("======================")
    print(f"Total positive patches saved: {total_patches}")


if __name__ == "__main__":
    main()