from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable, Optional, Tuple, List, Dict, Any

import numpy as np
import tifffile as tiff


# ============================================================
# Mosaic scan order (4x4 serpentine)
# ============================================================
TILE_ORDER_4x4 = np.array(
    [
        [1, 2, 3, 4],
        [8, 7, 6, 5],
        [9, 10, 11, 12],
        [16, 15, 14, 13],
    ],
    dtype=int,
)


# ============================================================
# Info container
# ============================================================
@dataclass
class MosaicInfo:
    tiles_dir: str
    tile_pattern: str
    ntiles_expected: int
    tiles_found: int
    tile_indices_found: list[int]
    tile_paths_ordered: list[str]
    tile_shape_example: tuple
    mosaic_shape: tuple
    mosaic_dtype: str
    order: str = "4x4_serpentine"


# ============================================================
# Index parsing (robust)
# ============================================================
# We accept several index tokens, in priority order:
#   ...t01... (your masks)
#   ...Im_00001...
#   ...tile_01...
#   ..._01... (fallback)
#
# Important: we search the full filename (stem), not only end.
_PATTERNS = [
    re.compile(r"(?:^|[_\-])t(?P<idx>\d{1,2})(?:$|[_\-])", re.IGNORECASE),
    re.compile(r"(?:^|[_\-])im[_\-]?(?P<idx>\d{1,5})(?:$|[_\-])", re.IGNORECASE),
    re.compile(r"(?:^|[_\-])tile[_\-]?(?P<idx>\d{1,5})(?:$|[_\-])", re.IGNORECASE),
    # fallback: any standalone _NN_ token
    re.compile(r"(?:^|[_\-])(?P<idx>\d{1,2})(?:$|[_\-])"),
]


def _parse_tile_index(path: Path) -> Optional[int]:
    """
    Extract tile index from filename (1..16 expected for 4x4).

    Examples that should work:
      ..._t01_mask.png -> 1
      Im_00016.tif -> 16
      tile_03.png -> 3
    """
    stem = path.stem  # no suffix
    # Normalize separators by adding underscores around
    # so patterns like ...t01... match safely.
    s = f"_{stem}_"

    for pat in _PATTERNS:
        m = pat.search(s)
        if m:
            try:
                idx = int(m.group("idx"))
            except Exception:
                continue
            return idx

    return None


# ============================================================
# Tile discovery
# ============================================================
def discover_tiles(tiles_dir: Path, tile_pattern: str, ntiles: int = 16) -> list[Path]:
    """
    Discover tile files and return ordered list of length ntiles.

    - Uses glob with tile_pattern.
    - Parses indices from filenames (supports 't01', 'Im_00001', etc.).
    - Validates presence of all 1..ntiles.
    """
    tiles_dir = Path(tiles_dir)
    paths = sorted(tiles_dir.glob(tile_pattern))
    if not paths:
        raise FileNotFoundError(f"No tiles matched pattern '{tile_pattern}' in: {tiles_dir}")

    idx_to_path: Dict[int, Path] = {}
    unparsable: List[Path] = []

    for p in paths:
        idx = _parse_tile_index(p)
        if idx is None:
            unparsable.append(p)
            continue
        # only keep indices in [1..ntiles], ignore others
        if 1 <= idx <= ntiles:
            # if duplicates exist, keep first (or newest). We'll keep first for stability.
            idx_to_path.setdefault(idx, p)

    if not idx_to_path:
        example = paths[0].name
        raise FileNotFoundError(
            f"Tiles matched '{tile_pattern}' but none had a parseable index (1..{ntiles}). "
            f"Example: {example}"
        )

    missing = [i for i in range(1, ntiles + 1) if i not in idx_to_path]
    if missing:
        # show a few parsed ones for debugging
        parsed = sorted(idx_to_path.keys())
        ex = paths[0].name
        raise FileNotFoundError(
            f"Missing tile indices: {missing}\n"
            f"Parsed indices found: {parsed}\n"
            f"Example filename: {ex}\n"
            f"Tip: expected tokens like 't01'...'t16' or 'Im_00001'...'Im_00016'."
        )

    return [idx_to_path[i] for i in range(1, ntiles + 1)]


# ============================================================
# Mosaic assembly
# ============================================================
def _assemble_mosaic_4x4(tiles: dict[int, np.ndarray]) -> np.ndarray:
    """
    Assemble 16 tiles into a single mosaic using TILE_ORDER_4x4 serpentine.

    Supports tiles as:
      - 2D (Y,X)
      - 3D (C,Y,X) or (Y,X,C) or (T,Y,X) etc.
    We will concatenate along Y and X keeping leading dims intact.
    """
    missing = [i for i in range(1, 17) if i not in tiles]
    if missing:
        raise FileNotFoundError(f"Missing tiles: {missing}")

    # Ensure all tiles same shape
    shapes = {tiles[i].shape for i in range(1, 17)}
    if len(shapes) != 1:
        raise ValueError(f"Tile shapes differ: {shapes}")

    tile_shape = next(iter(shapes))
    arr0 = tiles[1]
    ndim = arr0.ndim

    # We assume last two dims are Y,X
    if ndim < 2:
        raise ValueError(f"Tile must have at least 2 dims (Y,X). Got shape {tile_shape}")

    *lead, ty, tx = tile_shape
    mosaic_shape = (*lead, ty * 4, tx * 4)
    mosaic = np.zeros(mosaic_shape, dtype=arr0.dtype)

    for r in range(4):
        for c in range(4):
            idx = int(TILE_ORDER_4x4[r, c])
            y0, y1 = r * ty, (r + 1) * ty
            x0, x1 = c * tx, (c + 1) * tx
            mosaic[..., y0:y1, x0:x1] = tiles[idx]

    return mosaic


# ============================================================
# Public API
# ============================================================
def create_mosaic(
    tiles_dir: Path,
    tile_pattern: str = "Im_*.tif",
    ntiles: int = 16,
    loader: Optional[Callable[[Path], np.ndarray]] = None,
    dtype: Optional[np.dtype] = None,
) -> Tuple[np.ndarray, MosaicInfo]:
    """
    Generic mosaic builder.

    Parameters
    ----------
    tiles_dir : Path
        Directory containing the 16 tiles.
    tile_pattern : str
        Glob pattern, e.g. 'Im_*.tif' or '*_mask.png'
    ntiles : int
        Expected number of tiles (default 16 for 4x4).
    loader : callable
        Function Path->np.ndarray. If None:
          - uses tifffile for .tif/.tiff
          - uses imageio.v3 for others (png, jpg)
    dtype : np.dtype or None
        If provided, cast mosaic to dtype.

    Returns
    -------
    mosaic : np.ndarray
    info : MosaicInfo
    """
    tiles_dir = Path(tiles_dir)
    tile_paths = discover_tiles(tiles_dir, tile_pattern=tile_pattern, ntiles=ntiles)

    if loader is None:
        def _default_loader(p: Path) -> np.ndarray:
            suf = p.suffix.lower()
            if suf in (".tif", ".tiff"):
                return np.asarray(tiff.imread(str(p)))
            else:
                import imageio.v3 as iio
                return np.asarray(iio.imread(p))
        loader_fn = _default_loader
    else:
        loader_fn = loader

    tiles: Dict[int, np.ndarray] = {}
    for i, p in enumerate(tile_paths, start=1):
        arr = loader_fn(p)
        arr = np.asarray(arr)
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        tiles[i] = arr

    if ntiles != 16:
        raise NotImplementedError("Currently only supports 16 tiles (4x4).")

    mosaic = _assemble_mosaic_4x4(tiles)

    info = MosaicInfo(
        tiles_dir=str(tiles_dir),
        tile_pattern=str(tile_pattern),
        ntiles_expected=int(ntiles),
        tiles_found=int(len(tile_paths)),
        tile_indices_found=list(range(1, ntiles + 1)),
        tile_paths_ordered=[str(p) for p in tile_paths],
        tile_shape_example=tuple(tiles[1].shape),
        mosaic_shape=tuple(mosaic.shape),
        mosaic_dtype=str(mosaic.dtype),
    )

    return mosaic, info


def save_mosaic_tiff(out_path: Path, arr: np.ndarray, axes: str = "YX") -> None:
    """
    Save mosaic to TIFF with axes metadata.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(
        str(out_path),
        np.asarray(arr),
        photometric="minisblack",
        metadata={"axes": axes},
    )