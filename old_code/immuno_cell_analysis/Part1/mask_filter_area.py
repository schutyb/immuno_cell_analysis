from pathlib import Path
import numpy as np
import tifffile as tiff
from PIL import Image
import csv

# =========================================================
# CONFIG
# =========================================================
MIN_AREA_PX = 50  # filtra estructuras con area < MIN_AREA_PX

# =========================================================
# IO
# =========================================================
def load_mask(path: str) -> np.ndarray:
    """Load mask image (PNG/TIF/JPG) into a 2D numpy array."""
    path = Path(path)
    ext = path.suffix.lower()

    if ext in [".tif", ".tiff"]:
        arr = tiff.imread(str(path))
    else:
        arr = np.array(Image.open(str(path)))

    if arr.ndim == 3:
        arr = arr[..., 0]

    return arr

def save_mask_png(path: str, mask_uint8: np.ndarray):
    """
    Save binary mask as PNG.
    Expected values: 0 / 255
    """
    path = Path(path)
    img = Image.fromarray(mask_uint8.astype(np.uint8), mode="L")
    img.save(str(path))

def binarize(mask: np.ndarray, threshold: int = 0) -> np.ndarray:
    """Anything > threshold is foreground."""
    return mask > threshold

# =========================================================
# CONNECTED COMPONENTS (8-connectivity)
# =========================================================
def connected_components_8(mask_bool: np.ndarray):
    H, W = mask_bool.shape
    labels = np.zeros((H, W), dtype=np.int32)

    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 ( 0, -1),          ( 0, 1),
                 ( 1, -1), ( 1, 0), ( 1, 1)]

    current_label = 0
    areas = []

    for y in range(H):
        for x in range(W):
            if not mask_bool[y, x] or labels[y, x] != 0:
                continue

            current_label += 1
            stack = [(y, x)]
            labels[y, x] = current_label
            area = 0

            while stack:
                cy, cx = stack.pop()
                area += 1
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        if mask_bool[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current_label
                            stack.append((ny, nx))

            areas.append(area)

    return labels, areas

# =========================================================
# FILTER + CSV + FILTERED MASK
# =========================================================
def save_filtered_areas_csv(areas, min_area, csv_path):
    kept_ids = [i + 1 for i, a in enumerate(areas) if a >= min_area]

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["structure_id", "area_pixels"])
        for cid in kept_ids:
            writer.writerow([cid, int(areas[cid - 1])])

    return set(kept_ids)

def make_filtered_mask_from_labels(labels: np.ndarray, kept_ids: set) -> np.ndarray:
    if len(kept_ids) == 0:
        return np.zeros_like(labels, dtype=np.uint8)

    kept = np.isin(labels, list(kept_ids))
    return (kept.astype(np.uint8) * 255)

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    mask_path = "/Users/schutyb/Documents/balu_lab/immuno_cell_analysis_data/visit_04/Mosaic07_4x4_FOV600_z150_32Sp/mask_mosaic_4x4.png"
    out_csv = "/Users/schutyb/Documents/balu_lab/immuno_cell_analysis_data/visit_04/Mosaic07_4x4_FOV600_z150_32Sp/areas_filtered_min50px.csv"
    out_mask = "/Users/schutyb/Documents/balu_lab/immuno_cell_analysis_data/visit_04/Mosaic07_4x4_FOV600_z150_32Sp/mask_filtered_min50px.png" # máscara binaria final (PNG)

    raw = load_mask(mask_path)
    mask_bool = binarize(raw)

    labels, areas = connected_components_8(mask_bool)

    kept_ids = save_filtered_areas_csv(areas, MIN_AREA_PX, out_csv)
    filtered_mask = make_filtered_mask_from_labels(labels, kept_ids)

    save_mask_png(out_mask, filtered_mask)

    # ---- report ----
    total_area = int(mask_bool.sum())
    filtered_area = int((filtered_mask > 0).sum())
    removed = len(areas) - len(kept_ids)
    r_equiv = np.sqrt(MIN_AREA_PX / np.pi)

    print(f"Input mask: {mask_path}")
    print(f"MIN_AREA_PX: {MIN_AREA_PX} (radio equivalente ≈ {r_equiv:.2f} px)")
    print(f"Componentes totales: {len(areas)}")
    print(f"Componentes removidas: {removed}")
    print(f"Componentes mantenidas: {len(kept_ids)}")
    print(f"Área total (antes): {total_area} px")
    print(f"Área total (después): {filtered_area} px")
    print(f"✔ CSV guardado: {out_csv}")
    print(f"✔ Máscara filtrada PNG guardada: {out_mask}")
    