"""
create_mask_mosaics.py
======================

Create 4x4 serpentine mosaics from mask tiles stored as PNGs.

- Reads *_mask.png tiles
- Enforces binary masks (0/255)
- Saves ONLY:
    mask_mosaic_4x4.png

Accepted DATA_ROOT examples:
  1) project root with visit_*/Mosaic*/mask_*
  2) visit_01/
  3) Mosaic03_*/
  4) mask_* folder containing *_mask.png tiles
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import imageio.v3 as iio

from immuno_cell_analysis.utils.mosaic_utils import create_mosaic


# ============================================================
# CONFIG
# ============================================================
DATA_ROOT = Path("/Users/schutyb/Downloads/final masks-2/")
OUT_ROOT = DATA_ROOT / "masks_mosaic_out"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

TILE_PATTERN = "*_mask.png"
NTILES = 16
THRESHOLD = 0.5   # binarization threshold


# ============================================================
# Helpers
# ============================================================
def load_mask_tile(path: Path) -> np.ndarray:
    """
    Load mask tile and return binary uint8 (0/255).
    """
    arr = np.asarray(iio.imread(path))

    # ensure 2D
    if arr.ndim == 3:
        arr = arr[..., 0]

    arr = arr.astype(np.float32)
    vmax = arr.max() if arr.size else 0.0
    if vmax > 0:
        arr = arr / vmax

    mask = arr > THRESHOLD
    return (mask.astype(np.uint8) * 255)


def has_tiles(d: Path) -> bool:
    return d.is_dir() and any(d.glob(TILE_PATTERN))


def find_mask_dir(mosaic_dir: Path) -> Path | None:
    if has_tiles(mosaic_dir):
        return mosaic_dir
    for d in mosaic_dir.iterdir():
        if d.is_dir() and has_tiles(d):
            return d
    return None


def iter_jobs(root: Path):
    root = root.resolve()

    if has_tiles(root):
        parts = root.parts
        visit = next((p for p in parts if p.startswith("visit_")), "visit_unknown")
        mosaic = next((p for p in parts if p.startswith("Mosaic")), "Mosaic_unknown")
        yield visit, mosaic, root
        return

    if root.name.startswith("Mosaic"):
        md = find_mask_dir(root)
        if md:
            parts = root.parts
            visit = next((p for p in parts if p.startswith("visit_")), "visit_unknown")
            yield visit, root.name, md
        return

    if root.name.startswith("visit_"):
        for mdir in root.iterdir():
            if mdir.is_dir() and mdir.name.startswith("Mosaic"):
                md = find_mask_dir(mdir)
                if md:
                    yield root.name, mdir.name, md
        return

    for vdir in root.iterdir():
        if vdir.is_dir() and vdir.name.startswith("visit_"):
            for mdir in vdir.iterdir():
                if mdir.is_dir() and mdir.name.startswith("Mosaic"):
                    md = find_mask_dir(mdir)
                    if md:
                        yield vdir.name, mdir.name, md


# ============================================================
# Main
# ============================================================
def main():
    print("[INFO] Creating MASK mosaics (PNG only)")
    print(f"[INFO] DATA_ROOT: {DATA_ROOT}")
    print(f"[INFO] OUT_ROOT:  {OUT_ROOT}")

    jobs = list(iter_jobs(DATA_ROOT))
    if not jobs:
        raise FileNotFoundError("No mask tiles found.")

    for visit, mosaic, mask_dir in jobs:
        out_dir = OUT_ROOT / visit / mosaic
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[VISIT] {visit} | [MOSAIC] {mosaic}")
        print(f"  mask tiles: {mask_dir}")

        mosaic_img, _ = create_mosaic(
            tiles_dir=mask_dir,
            tile_pattern=TILE_PATTERN,
            ntiles=NTILES,
            loader=load_mask_tile,
            dtype=np.uint8,
        )

        # ensure binary at mosaic level
        mosaic_img = (mosaic_img > 0).astype(np.uint8) * 255

        out_png = out_dir / "mask_mosaic_4x4.png"
        iio.imwrite(out_png, mosaic_img)

        print(f"  [SAVED] {out_png}")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()