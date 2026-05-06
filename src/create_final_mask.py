#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

from PIL import Image
from skimage.measure import label, regionprops
from skimage.transform import resize
from sklearn.mixture import GaussianMixture


# =========================
# CONFIG
# =========================

PATIENT_DIR = Path(
    "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449"
).expanduser()

PHASOR_SUBDIR = "phasor"
PHASOR_NAME = "phasor_calibrated_green_blue_mosaic.tif"

SEG_SUBDIR = "SegData"

OUTPUT_SUBDIR = "segmentation_area_phasor"

MIN_AREA_PX = 15

# Usamos green detector para clasificación lifetime por defecto
# phasor planes:
# 0 = DC
# 1 = G green
# 2 = S green
# 3 = G blue
# 4 = S blue
G_PLANE = 1
S_PLANE = 2

# visitas con 3 familias: melanina, células, elastina
N_COMPONENTS_BY_VISIT = {
    "visit01": 3,
    "visit02": 3,
    "visit03": 3,
    "visit04": 2,
}

DEFAULT_N_COMPONENTS = 3

RANDOM_STATE = 0

SAVE_DEBUG_CLASS_MASKS = True


# =========================
# HELPERS
# =========================

def natural_key(path):
    return [
        int(t) if t.isdigit() else t.lower()
        for t in re.split(r"(\d+)", str(path.name))
    ]


def parse_mosaic_shape_from_name(folder_name):
    m = re.search(r"(\d+)x(\d+)", folder_name)
    if not m:
        raise ValueError(f"No pude detectar forma del mosaico en: {folder_name}")
    return int(m.group(1)), int(m.group(2))


def extract_tile_number(name):
    patterns = [
        r"Im_(\d+)",
        r"_t(\d+)",
        r"tile[_-]?(\d+)",
    ]

    for pat in patterns:
        m = re.search(pat, name, re.IGNORECASE)
        if m:
            return int(m.group(1))

    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else None


def collect_mask_paths(seg_dir):
    files = []
    for ext in ("*.tif", "*.tiff", "*.png"):
        files.extend(seg_dir.glob(ext))

    pairs = []
    for p in files:
        tile = extract_tile_number(p.name)
        if tile is not None:
            pairs.append((tile, p))

    pairs.sort(key=lambda x: x[0])
    return [p for _, p in pairs]


def read_mask(path):
    arr = np.array(Image.open(path))

    if arr.ndim == 3:
        arr = arr[..., 0]

    return arr > 0


def snake_indices(nrows, ncols):
    idx = np.arange(nrows * ncols).reshape(nrows, ncols)
    layout = []

    for r in range(nrows):
        row = idx[r].copy()
        if r % 2 == 1:
            row = row[::-1]
        layout.append(row.tolist())

    return layout


def stitch_tiles_snake(tile_imgs, nrows, ncols):
    expected = nrows * ncols
    if len(tile_imgs) != expected:
        raise ValueError(f"Esperaba {expected} tiles, recibí {len(tile_imgs)}")

    h, w = tile_imgs[0].shape
    mosaic = np.zeros((nrows * h, ncols * w), dtype=tile_imgs[0].dtype)

    layout = snake_indices(nrows, ncols)

    for r in range(nrows):
        for c in range(ncols):
            tile_idx = layout[r][c]

            y0 = r * h
            y1 = y0 + h
            x0 = c * w
            x1 = x0 + w

            mosaic[y0:y1, x0:x1] = tile_imgs[tile_idx]

    return mosaic


def split_mosaic_to_tiles_snake(mosaic, nrows, ncols):
    h_total, w_total = mosaic.shape
    h = h_total // nrows
    w = w_total // ncols

    tiles = [None] * (nrows * ncols)
    layout = snake_indices(nrows, ncols)

    for r in range(nrows):
        for c in range(ncols):
            tile_idx = layout[r][c]

            y0 = r * h
            y1 = y0 + h
            x0 = c * w
            x1 = x0 + w

            tiles[tile_idx] = mosaic[y0:y1, x0:x1]

    return tiles


def resize_mask_if_needed(mask, target_shape):
    if mask.shape == target_shape:
        return mask

    mask_rs = resize(
        mask.astype(float),
        target_shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )

    return mask_rs > 0.5


def phase_deg(g, s):
    return np.degrees(np.arctan2(s, g)).astype(np.float32)


def get_visit_name(mosaic_dir):
    return mosaic_dir.parent.name


def get_n_components(visit_name):
    return N_COMPONENTS_BY_VISIT.get(visit_name, DEFAULT_N_COMPONENTS)


def assign_gmm_classes(df, n_components):
    """
    Asigna clases biológicas por phase promedio:
    menor phase -> melanina
    intermedia -> células
    mayor phase -> elastina

    Para 2 componentes:
    menor phase -> células
    mayor phase -> elastina
    """
    cluster_summary = (
        df.groupby("gmm_cluster")
        .agg(
            mean_phase_deg=("mean_phase_deg", "mean"),
            mean_g=("mean_g", "mean"),
            mean_s=("mean_s", "mean"),
            n_rois=("roi_id", "count"),
        )
        .reset_index()
        .sort_values("mean_phase_deg")
    )

    ordered_clusters = cluster_summary["gmm_cluster"].tolist()

    if n_components == 3:
        class_map = {
            ordered_clusters[0]: "melanin",
            ordered_clusters[1]: "cell",
            ordered_clusters[2]: "elastin",
        }
    elif n_components == 2:
        class_map = {
            ordered_clusters[0]: "cell",
            ordered_clusters[1]: "elastin",
        }
    else:
        raise ValueError("Solo está implementado GMM con 2 o 3 componentes.")

    df["lifetime_class"] = df["gmm_cluster"].map(class_map)

    cluster_summary["lifetime_class"] = cluster_summary["gmm_cluster"].map(class_map)

    return df, cluster_summary


def plot_gmm_phasor(df, cluster_summary, out_path, title):
    colors = {
        "melanin": "tab:brown",
        "cell": "tab:green",
        "elastin": "tab:orange",
        "unknown": "tab:gray",
    }

    plt.figure(figsize=(7, 6))

    for cls, sub in df.groupby("lifetime_class"):
        plt.scatter(
            sub["mean_g"],
            sub["mean_s"],
            s=10,
            alpha=0.65,
            label=f"{cls} (n={len(sub)})",
            color=colors.get(cls, "tab:gray"),
        )

    for _, row in cluster_summary.iterrows():
        plt.scatter(
            row["mean_g"],
            row["mean_s"],
            s=160,
            marker="x",
            linewidths=3,
            color=colors.get(row["lifetime_class"], "black"),
        )

        plt.text(
            row["mean_g"],
            row["mean_s"],
            f" {row['lifetime_class']}",
            fontsize=9,
            va="center",
        )

    theta = np.linspace(0, np.pi, 400)
    gx = 0.5 + 0.5 * np.cos(theta)
    sy = 0.5 * np.sin(theta)
    plt.plot(gx, sy, "k--", linewidth=1, alpha=0.5)

    plt.xlabel("G")
    plt.ylabel("S")
    plt.title(title)
    plt.axis("equal")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()


# =========================
# PROCESS MOSAIC
# =========================

def process_mosaic(mosaic_dir):
    visit_name = get_visit_name(mosaic_dir)
    mosaic_name = mosaic_dir.name

    print(f"\n[PROCESS] {visit_name} | {mosaic_name}")

    phasor_path = mosaic_dir / PHASOR_SUBDIR / PHASOR_NAME
    seg_dir = mosaic_dir / SEG_SUBDIR
    out_dir = mosaic_dir / OUTPUT_SUBDIR

    out_dir.mkdir(parents=True, exist_ok=True)

    if not phasor_path.exists():
        print(f"[SKIP] No existe phasor: {phasor_path}")
        return

    if not seg_dir.exists():
        print(f"[SKIP] No existe SegData: {seg_dir}")
        return

    phasor = tiff.imread(phasor_path).astype(np.float32)

    if phasor.ndim != 3 or phasor.shape[0] < 5:
        raise ValueError(f"Phasor inválido: {phasor.shape}")

    dc = phasor[0]
    g = phasor[G_PLANE]
    s = phasor[S_PLANE]

    nrows, ncols = parse_mosaic_shape_from_name(mosaic_name)
    expected_tiles = nrows * ncols

    mask_paths = collect_mask_paths(seg_dir)

    if len(mask_paths) != expected_tiles:
        raise ValueError(
            f"{mosaic_name}: esperaba {expected_tiles} máscaras, "
            f"encontré {len(mask_paths)}"
        )

    mask_tiles = [read_mask(p) for p in mask_paths]
    mask_mosaic = stitch_tiles_snake(mask_tiles, nrows, ncols)

    mask_mosaic = resize_mask_if_needed(mask_mosaic, dc.shape)

    # Label global del mosaico
    labeled = label(mask_mosaic, connectivity=2)

    rows = []
    keep_area_mask = np.zeros_like(mask_mosaic, dtype=bool)

    roi_id = 0

    for prop in regionprops(labeled):
        area = prop.area

        if area < MIN_AREA_PX:
            continue

        roi_mask = labeled == prop.label

        roi_g = g[roi_mask]
        roi_s = s[roi_mask]
        roi_dc = dc[roi_mask]

        mean_g = np.nanmean(roi_g)
        mean_s = np.nanmean(roi_s)
        mean_dc = np.nanmean(roi_dc)

        if not np.isfinite(mean_g) or not np.isfinite(mean_s):
            continue

        roi_id += 1

        mean_phase = float(phase_deg(mean_g, mean_s))
        mean_modulation = float(np.sqrt(mean_g**2 + mean_s**2))

        rows.append(
            {
                "visit": visit_name,
                "mosaic": mosaic_name,
                "roi_id": roi_id,
                "label_original": prop.label,
                "area_px": float(area),
                "centroid_y": float(prop.centroid[0]),
                "centroid_x": float(prop.centroid[1]),
                "mean_dc": float(mean_dc),
                "mean_g": float(mean_g),
                "mean_s": float(mean_s),
                "mean_phase_deg": mean_phase,
                "mean_modulation": mean_modulation,
            }
        )

        keep_area_mask[roi_mask] = True

    if len(rows) == 0:
        print("[WARNING] No quedaron ROIs después del filtro de área.")
        return

    df = pd.DataFrame(rows)

    n_components = get_n_components(visit_name)

    if len(df) < n_components:
        print(
            f"[WARNING] ROIs insuficientes para GMM: n_rois={len(df)}, "
            f"n_components={n_components}"
        )
        return

    X = df[["mean_g", "mean_s"]].to_numpy(dtype=np.float32)

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=RANDOM_STATE,
    )

    df["gmm_cluster"] = gmm.fit_predict(X)
    df, cluster_summary = assign_gmm_classes(df, n_components=n_components)

    # Guardar tabla ROI
    roi_csv = out_dir / f"{mosaic_name}_roi_area_phasor_gmm.csv"
    df.to_csv(roi_csv, index=False)

    # Guardar resumen GMM
    gmm_csv = out_dir / f"{mosaic_name}_gmm_cluster_summary.csv"
    cluster_summary.to_csv(gmm_csv, index=False)

    # Guardar centro de masa / distribución elastina
    elastin = df[df["lifetime_class"] == "elastin"]

    if len(elastin) > 0:
        elastin_summary = pd.DataFrame(
            [
                {
                    "visit": visit_name,
                    "mosaic": mosaic_name,
                    "n_elastin_rois": len(elastin),
                    "elastin_cm_g": float(np.nanmean(elastin["mean_g"])),
                    "elastin_cm_s": float(np.nanmean(elastin["mean_s"])),
                    "elastin_cm_phase_deg": float(np.nanmean(elastin["mean_phase_deg"])),
                    "elastin_cm_modulation": float(np.nanmean(elastin["mean_modulation"])),
                    "elastin_std_g": float(np.nanstd(elastin["mean_g"])),
                    "elastin_std_s": float(np.nanstd(elastin["mean_s"])),
                    "elastin_std_phase_deg": float(np.nanstd(elastin["mean_phase_deg"])),
                }
            ]
        )
    else:
        elastin_summary = pd.DataFrame(
            [
                {
                    "visit": visit_name,
                    "mosaic": mosaic_name,
                    "n_elastin_rois": 0,
                    "elastin_cm_g": np.nan,
                    "elastin_cm_s": np.nan,
                    "elastin_cm_phase_deg": np.nan,
                    "elastin_cm_modulation": np.nan,
                    "elastin_std_g": np.nan,
                    "elastin_std_s": np.nan,
                    "elastin_std_phase_deg": np.nan,
                }
            ]
        )

    elastin_csv = out_dir / f"{mosaic_name}_elastin_cm.csv"
    elastin_summary.to_csv(elastin_csv, index=False)

    # Crear máscaras finales
    final_cell_mask = np.zeros_like(mask_mosaic, dtype=np.uint8)
    final_melanin_mask = np.zeros_like(mask_mosaic, dtype=np.uint8)
    final_elastin_mask = np.zeros_like(mask_mosaic, dtype=np.uint8)
    final_all_area_lifetime_mask = np.zeros_like(mask_mosaic, dtype=np.uint16)

    class_to_value = {
        "melanin": 1,
        "cell": 2,
        "elastin": 3,
    }

    # Reconstruimos a partir del label original
    for _, row in df.iterrows():
        roi_mask = labeled == int(row["label_original"])
        cls = row["lifetime_class"]

        final_all_area_lifetime_mask[roi_mask] = class_to_value.get(cls, 0)

        if cls == "cell":
            final_cell_mask[roi_mask] = 1
        elif cls == "melanin":
            final_melanin_mask[roi_mask] = 1
        elif cls == "elastin":
            final_elastin_mask[roi_mask] = 1

    # Guardar máscaras mosaico
    tiff.imwrite(out_dir / f"{mosaic_name}_mask_area_filtered.tif", keep_area_mask.astype(np.uint8))
    tiff.imwrite(out_dir / f"{mosaic_name}_mask_lifetime_classes.tif", final_all_area_lifetime_mask)
    tiff.imwrite(out_dir / f"{mosaic_name}_cell_mask_final.tif", final_cell_mask.astype(np.uint8))

    if SAVE_DEBUG_CLASS_MASKS:
        tiff.imwrite(out_dir / f"{mosaic_name}_melanin_mask.tif", final_melanin_mask.astype(np.uint8))
        tiff.imwrite(out_dir / f"{mosaic_name}_elastin_mask.tif", final_elastin_mask.astype(np.uint8))

    # Guardar máscaras finales por tile
    tile_out_dir = out_dir / "tiles"
    tile_out_dir.mkdir(parents=True, exist_ok=True)

    cell_tiles = split_mosaic_to_tiles_snake(final_cell_mask, nrows, ncols)
    class_tiles = split_mosaic_to_tiles_snake(final_all_area_lifetime_mask, nrows, ncols)

    for mask_path, cell_tile, class_tile in zip(mask_paths, cell_tiles, class_tiles):
        tile_num = extract_tile_number(mask_path.name)
        tile_label = f"{tile_num:05d}" if tile_num is not None else mask_path.stem

        tiff.imwrite(
            tile_out_dir / f"Im_{tile_label}_cell_mask_final.tif",
            cell_tile.astype(np.uint8),
        )

        tiff.imwrite(
            tile_out_dir / f"Im_{tile_label}_lifetime_classes.tif",
            class_tile.astype(np.uint16),
        )

    # Plot GMM phasor
    gmm_plot = out_dir / f"{mosaic_name}_phasor_gmm.png"
    plot_gmm_phasor(
        df=df,
        cluster_summary=cluster_summary,
        out_path=gmm_plot,
        title=f"{visit_name} | {mosaic_name} | ROI mean phasor GMM",
    )

    print(f"[OK] ROIs after area filter: {len(df)}")
    print(f"     ROI CSV: {roi_csv}")
    print(f"     Elastin CM: {elastin_csv}")
    print(f"     Final cell mask: {out_dir / f'{mosaic_name}_cell_mask_final.tif'}")
    print(f"     GMM plot: {gmm_plot}")


# =========================
# MAIN
# =========================

def main():
    visit_dirs = sorted(
        [p for p in PATIENT_DIR.glob("visit*") if p.is_dir()],
        key=natural_key,
    )

    for visit_dir in visit_dirs:
        mosaic_dirs = sorted(
            [p for p in visit_dir.glob("Mosaic*") if p.is_dir()],
            key=natural_key,
        )

        for mosaic_dir in mosaic_dirs:
            try:
                process_mosaic(mosaic_dir)
            except Exception as e:
                print(f"[ERROR] {mosaic_dir}")
                print(f"        {type(e).__name__}: {e}")

    print("\nListo.")


if __name__ == "__main__":
    main()