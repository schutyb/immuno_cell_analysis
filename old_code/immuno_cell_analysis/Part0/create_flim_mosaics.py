from __future__ import annotations

from pathlib import Path
import numpy as np
import tifffile as tiff

from immuno_cell_analysis.utils.mosaic_utils import create_mosaic, save_mosaic_tiff


# ============================================================
# CONFIG  ✅ editá esto
# ============================================================
DATA_ROOT = Path("/Users/schutyb/Documents/balu_lab/immuno_cell_analysis_data")
OUT_ROOT  = DATA_ROOT / "flim_mosaic_out"

# dónde están los tiles FLIM dentro de cada Mosaic*
# (por tu workflow: visit_XX/Mosaic.../Im_00001.tif ... Im_00016.tif)
TILE_PATTERN = "Im_*.tif"

# si querés castear para achicar tamaño (opcional)
CAST_DTYPE = np.float32  # None para no castear

# ============================================================
# Helpers
# ============================================================
def find_visits(root: Path) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("visit_")])


def find_mosaics(visit_dir: Path) -> list[Path]:
    return sorted([p for p in visit_dir.iterdir() if p.is_dir() and p.name.startswith("Mosaic")])


def robust_tiff_loader(p: Path) -> np.ndarray:
    """
    Loader robusto para tiles FLIM:
    - intenta tiff.imread normal
    - si hay series raras, stackea páginas
    """
    try:
        arr = np.asarray(tiff.imread(str(p)))
    except Exception:
        with tiff.TiffFile(str(p)) as tf:
            pages = [pg.asarray() for pg in tf.pages]
        arr = np.stack(pages, axis=0)

    # no tocamos el orden de ejes: create_mosaic concatena las últimas 2 dims (Y,X)
    return np.asarray(arr)


def infer_axes(arr: np.ndarray) -> str:
    """
    Devuelve un string de ejes para metadata.
    Asumimos que las últimas 2 dims son YX.
    """
    if arr.ndim == 2:
        return "YX"
    if arr.ndim == 3:
        # típicamente (T,Y,X) o (Z,Y,X) o (C,Y,X)
        return "TYX"
    if arr.ndim == 4:
        # ej (C,T,Y,X) o (Z,T,Y,X) etc.
        return "CTYX"
    # fallback genérico:
    return "..."


# ============================================================
# Main
# ============================================================
def main():
    print("[INFO] Creating FLIM mosaics for ALL visits...")
    print(f"[INFO] DATA_ROOT: {DATA_ROOT}")
    print(f"[INFO] OUT_ROOT:  {OUT_ROOT}")

    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"DATA_ROOT not found: {DATA_ROOT}")

    visits = find_visits(DATA_ROOT)
    if not visits:
        raise FileNotFoundError(f"No visit_* folders under: {DATA_ROOT}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    n_done = 0
    n_skip = 0

    for visit_dir in visits:
        mosaics = find_mosaics(visit_dir)
        if not mosaics:
            print(f"[WARN] {visit_dir.name}: no Mosaic* folders found")
            continue

        for mosaic_dir in mosaics:
            print(f"\n[VISIT] {visit_dir.name} | [MOSAIC] {mosaic_dir.name}")
            tiles_dir = mosaic_dir  # acá están los Im_00001.tif ... Im_00016.tif

            try:
                mosaic_img, info = create_mosaic(
                    tiles_dir=tiles_dir,
                    tile_pattern=TILE_PATTERN,
                    ntiles=16,
                    loader=robust_tiff_loader,
                    dtype=CAST_DTYPE,
                )
            except Exception as e:
                print(f"  [SKIP] Could not create mosaic: {e}")
                n_skip += 1
                continue

            # output path (estructura simple por visita/mosaico)
            out_dir = OUT_ROOT / visit_dir.name / mosaic_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)

            axes = infer_axes(mosaic_img)
            out_path = out_dir / "flim_mosaic.tif"
            save_mosaic_tiff(out_path, mosaic_img, axes=axes)

            # opcional: guardar info
            (out_dir / "mosaic_info.txt").write_text(
                "\n".join([
                    f"tiles_dir={info.tiles_dir}",
                    f"tile_pattern={info.tile_pattern}",
                    f"tiles_found={info.tiles_found}",
                    f"tile_shape_example={info.tile_shape_example}",
                    f"mosaic_shape={info.mosaic_shape}",
                    f"mosaic_dtype={info.mosaic_dtype}",
                    f"order={info.order}",
                ]) + "\n"
            )

            print(f"  [OK] saved: {out_path}")
            print(f"       shape: {mosaic_img.shape} | dtype: {mosaic_img.dtype} | axes: {axes}")
            n_done += 1

    print("\n✅ DONE")
    print(f"[INFO] mosaics saved: {n_done}")
    print(f"[INFO] skipped:      {n_skip}")
    print(f"[INFO] outputs in:   {OUT_ROOT}")


if __name__ == "__main__":
    main()