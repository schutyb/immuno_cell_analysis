"""Minimal FLIM TIFF discovery, reading, and detector-splitting helpers."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import tifffile

ACCEPTED_INPUT_BINS = (31, 32)


def natural_key(value: str | Path) -> tuple:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", str(value))
    )


def find_flim_directories(patient_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in patient_dir.rglob("*")
            if path.is_dir() and path.name.casefold() == "flim"
        ),
        key=natural_key,
    )


def detector_type_from_mosaic_name(name: str) -> str | None:
    if re.search(r"sp$", name, flags=re.IGNORECASE):
        return "split"
    if re.search(r"a1$", name, flags=re.IGNORECASE):
        return "green"
    if re.search(r"a0$", name, flags=re.IGNORECASE):
        return "blue"
    return None


def single_detector_pair_key(mosaic_dir: Path) -> tuple[str, str]:
    normalized = re.sub(r"^mosaic\d+", "mosaic", mosaic_dir.name, flags=re.IGNORECASE)
    normalized = re.sub(r"a[01]$", "a", normalized, flags=re.IGNORECASE)
    return str(mosaic_dir.parent.resolve()).casefold(), normalized.casefold()


def classify_input_directories(
    flim_dirs: list[Path],
) -> tuple[list[Path], list[tuple[Path, Path]], list[str]]:
    """Return split folders, paired (green A1, blue A0) folders, and warnings."""
    split_dirs: list[Path] = []
    grouped: dict[tuple[str, str], dict[str, list[Path]]] = {}
    warnings: list[str] = []
    for flim_dir in flim_dirs:
        detector = detector_type_from_mosaic_name(flim_dir.parent.name)
        if detector == "split":
            split_dirs.append(flim_dir)
        elif detector in {"green", "blue"}:
            key = single_detector_pair_key(flim_dir.parent)
            grouped.setdefault(key, {"green": [], "blue": []})[detector].append(
                flim_dir
            )
        else:
            warnings.append(
                "Carpeta flim ignorada porque el mosaico no termina en "
                f"Sp, A1 o A0: {flim_dir}"
            )

    pairs: list[tuple[Path, Path]] = []
    for key, channels in sorted(grouped.items(), key=lambda item: item[0]):
        green_dirs = sorted(channels["green"], key=natural_key)
        blue_dirs = sorted(channels["blue"], key=natural_key)
        pair_count = min(len(green_dirs), len(blue_dirs))
        pairs.extend(zip(green_dirs[:pair_count], blue_dirs[:pair_count]))
        for unmatched in green_dirs[pair_count:]:
            warnings.append(f"Adquisición A1 sin A0 correspondiente: {unmatched}")
        for unmatched in blue_dirs[pair_count:]:
            warnings.append(f"Adquisición A0 sin A1 correspondiente: {unmatched}")
        if pair_count > 1:
            warnings.append(
                f"La clave {key[1]!r} produjo {pair_count} pares A1/A0; "
                "se emparejaron por orden natural."
            )
    return (
        sorted(split_dirs, key=natural_key),
        sorted(pairs, key=lambda pair: natural_key(pair[0])),
        warnings,
    )


def extract_tile_number(path: Path) -> int | None:
    match = re.fullmatch(r"Im_(\d+)\.(?:tif|tiff)", path.name, flags=re.IGNORECASE)
    return None if match is None else int(match.group(1))


def find_tiles(flim_dir: Path) -> list[Path]:
    numbered: list[tuple[int, Path]] = []
    for path in flim_dir.iterdir():
        if not path.is_file():
            continue
        number = extract_tile_number(path)
        if number is not None:
            numbered.append((number, path))
    numbered.sort(key=lambda item: item[0])
    numbers = [number for number, _ in numbered]
    if len(numbers) != len(set(numbers)):
        raise ValueError(f"Duplicate tile numbers in {flim_dir}: {numbers}")
    return [path for _, path in numbered]


def read_tiff_robust(path: Path) -> np.ndarray:
    """Read physical pages when malformed shaped metadata cannot be trusted."""
    try:
        with tifffile.TiffFile(path) as tif:
            if not tif.pages:
                raise ValueError("TIFF contains no pages")
            page_shape = tuple(int(value) for value in tif.pages[0].shape)
            physical_elements = len(tif.pages) * int(np.prod(page_shape))
            declared_elements = int(np.prod(tif.series[0].shape))
            if declared_elements != physical_elements:
                return tif.asarray(key=range(len(tif.pages)))
        return tifffile.imread(path)
    except Exception as standard_error:
        try:
            with tifffile.TiffFile(path) as tif:
                pages = [page.asarray() for page in tif.pages]
            if not pages:
                raise ValueError("TIFF contains no pages")
            if len(pages) == 1:
                return pages[0]
            shape, dtype = pages[0].shape, pages[0].dtype
            if any(page.shape != shape or page.dtype != dtype for page in pages):
                raise ValueError("TIFF pages do not share shape and dtype")
            return np.stack(pages, axis=0)
        except Exception as fallback_error:
            raise RuntimeError(
                f"Could not read {path}; standard={standard_error}; "
                f"page_fallback={fallback_error}"
            ) from fallback_error


def detect_bin_axis(image: np.ndarray) -> int:
    candidates = [
        axis for axis, size in enumerate(image.shape) if size in ACCEPTED_INPUT_BINS
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Cannot identify one bin axis in shape {image.shape}; "
            f"candidates={candidates}"
        )
    return candidates[0]


def prepare_tile(tile_path: Path) -> tuple[np.ndarray, tuple[int, ...], int]:
    raw = read_tiff_robust(tile_path)
    original_shape = raw.shape
    if raw.ndim != 3:
        raise ValueError(f"Expected a 3-D FLIM TIFF, got {raw.shape}: {tile_path}")
    bin_axis = detect_bin_axis(raw)
    tile = np.moveaxis(raw, bin_axis, -1)
    return tile, original_shape, bin_axis


def split_green_blue(tile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a 31/32-bin simultaneous acquisition into two corrected 16 bins."""
    if tile.ndim != 3 or tile.shape[-1] not in ACCEPTED_INPUT_BINS:
        raise ValueError(f"Expected Y,X,31/32 split tile, got {tile.shape}")
    if tile.shape[-1] == 31:
        tile = np.concatenate([tile, tile[..., -1:]], axis=-1)
    green = tile[..., :16].copy()
    blue = tile[..., 16:32].copy()
    green[..., -1] = green[..., -2]
    blue[..., -1] = blue[..., -2]
    return green, blue


def normalize_single_detector_32_bins(tile: np.ndarray) -> np.ndarray:
    """Normalize A0/A1 to 32 bins and copy bin 31 into bin 32."""
    if tile.ndim != 3 or tile.shape[-1] not in ACCEPTED_INPUT_BINS:
        raise ValueError(f"Expected Y,X,31/32 A0/A1 tile, got {tile.shape}")
    if tile.shape[-1] == 31:
        return np.concatenate([tile, tile[..., 30:31]], axis=-1)
    normalized = tile.copy()
    normalized[..., 31] = normalized[..., 30]
    return normalized
