import re
from pathlib import Path
import numpy as np
import tifffile as tiff
from PIL import Image


PATIENT_FOLDER = "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449"

PATCH_SIZE = 128
STRIDE = 128

TARGET_EMPTY_PATCHES = 1300
RANDOM_SEED = 0

MAX_BLACK_FRACTION = 0.30
BLACK_THRESHOLD = 5

CLEAR_OLD_EMPTY_PATCHES = True


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


def get_tile_id(path):
    match = re.search(r"(Im_\d+)", path.stem)
    if match is None:
        raise ValueError(f"Could not detect tile ID from: {path.name}")
    return match.group(1)


def get_rgb_for_tile(random_forest_dir, tile_id):
    rgb_dir = random_forest_dir / "rgb"

    candidates = [
        rgb_dir / f"{tile_id}_pseudoRGB.tif",
        rgb_dir / f"{tile_id}_pseudoRGB.tiff",
        rgb_dir / f"{tile_id}_pseudoRGB.png",
    ]

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(f"No RGB found for {tile_id} in {rgb_dir}")


def is_bad_patch(img_patch):
    black_fraction = np.mean(np.all(img_patch < BLACK_THRESHOLD, axis=-1))
    return black_fraction > MAX_BLACK_FRACTION


def find_random_forest_dirs(patient_folder):
    return sorted([
        p for p in Path(patient_folder).glob("visit*/Mosaic*/random_forest")
        if p.is_dir()
    ])


def group_masks_by_tile(random_forest_dir):
    mask_dir = random_forest_dir / "mask"
    if not mask_dir.exists():
        return {}

    mask_files = sorted(list(mask_dir.glob("*.tif")) + list(mask_dir.glob("*.tiff")))

    grouped = {}
    for mask_path in mask_files:
        try:
            tile_id = get_tile_id(mask_path)
            grouped.setdefault(tile_id, []).append(mask_path)
        except Exception as e:
            print(f"Skipping mask {mask_path.name}: {e}")

    return grouped


def combine_masks(mask_paths):
    combined = None

    for p in mask_paths:
        mask = load_mask(p)

        if combined is None:
            combined = mask.copy()
        else:
            if mask.shape != combined.shape:
                raise ValueError(f"Mask shape mismatch: {p}")
            combined = np.logical_or(combined, mask).astype(np.uint8)

    return combined.astype(np.uint8)


def clear_old_empty_outputs(random_forest_dirs):
    if not CLEAR_OLD_EMPTY_PATCHES:
        return

    for rf_dir in random_forest_dirs:
        patch_dir = rf_dir / "patch"
        mask_dir = rf_dir / "patch_mask"

        if patch_dir.exists():
            for f in patch_dir.glob("*_empty*.png"):
                f.unlink()

        if mask_dir.exists():
            for f in mask_dir.glob("*_empty*_mask.png"):
                f.unlink()


def collect_empty_candidates(random_forest_dirs):
    candidates = []

    for rf_dir in random_forest_dirs:
        grouped = group_masks_by_tile(rf_dir)

        if len(grouped) == 0:
            continue

        mosaic_dir = rf_dir.parent
        visit_dir = mosaic_dir.parent

        for tile_id, mask_paths in grouped.items():
            try:
                rgb_path = get_rgb_for_tile(rf_dir, tile_id)
                img = load_rgb(rgb_path)
                mask = combine_masks(mask_paths)

                if img.shape[:2] != mask.shape:
                    print(f"Skipping {tile_id}: shape mismatch")
                    continue

                h, w = mask.shape

                for y in range(0, h - PATCH_SIZE + 1, STRIDE):
                    for x in range(0, w - PATCH_SIZE + 1, STRIDE):

                        mask_patch = mask[y:y + PATCH_SIZE, x:x + PATCH_SIZE]

                        # only patches with zero annotated cells
                        if mask_patch.sum() > 0:
                            continue

                        img_patch = img[y:y + PATCH_SIZE, x:x + PATCH_SIZE]

                        if is_bad_patch(img_patch):
                            continue

                        prefix = f"{visit_dir.name}_{mosaic_dir.name}_{tile_id}_empty_y{y:04d}_x{x:04d}"

                        candidates.append({
                            "rf_dir": rf_dir,
                            "rgb_path": rgb_path,
                            "tile_id": tile_id,
                            "visit": visit_dir.name,
                            "mosaic": mosaic_dir.name,
                            "x": x,
                            "y": y,
                            "prefix": prefix,
                        })

            except Exception as e:
                print(f"ERROR collecting candidates in {rf_dir} {tile_id}: {e}")

    return candidates


def save_empty_patch(candidate, idx):
    rf_dir = candidate["rf_dir"]
    rgb_path = candidate["rgb_path"]
    x = candidate["x"]
    y = candidate["y"]

    patch_dir = rf_dir / "patch"
    patch_mask_dir = rf_dir / "patch_mask"

    patch_dir.mkdir(parents=True, exist_ok=True)
    patch_mask_dir.mkdir(parents=True, exist_ok=True)

    img = load_rgb(rgb_path)

    img_patch = img[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
    empty_mask = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)

    name = f"{candidate['prefix']}_empty{idx:04d}.png"
    mask_name = f"{candidate['prefix']}_empty{idx:04d}_mask.png"

    Image.fromarray(img_patch).save(patch_dir / name)
    Image.fromarray(empty_mask).save(patch_mask_dir / mask_name)


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    rf_dirs = find_random_forest_dirs(PATIENT_FOLDER)

    print(f"random_forest folders found: {len(rf_dirs)}")

    clear_old_empty_outputs(rf_dirs)

    candidates = collect_empty_candidates(rf_dirs)

    print(f"Eligible empty patch candidates: {len(candidates)}")

    if len(candidates) == 0:
        raise RuntimeError("No empty patch candidates found.")

    n_to_save = min(TARGET_EMPTY_PATCHES, len(candidates))

    selected_idx = rng.choice(len(candidates), size=n_to_save, replace=False)

    for out_idx, cand_idx in enumerate(selected_idx, start=1):
        save_empty_patch(candidates[cand_idx], out_idx)

    print("\nDone.")
    print(f"Empty patches saved: {n_to_save}")


if __name__ == "__main__":
    main()