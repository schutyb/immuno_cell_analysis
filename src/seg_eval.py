#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Evaluate Unet of PC and final area+phasor segmentation against manual masks.

from pathlib import Path
import re
import numpy as np
import pandas as pd
import tifffile as tiff
from PIL import Image
from skimage.measure import label
from skimage.transform import resize


# =========================
# CONFIG
# =========================

PATIENT_DIR = Path(
    "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449"
).expanduser()

MANUAL_MASK_SUBDIR = Path("random_forest/mask")
PETER_MASK_SUBDIR = Path("SegData")
FINAL_MASK_SUBDIR = Path("segmentation_area_phasor/tiles")

OUTPUT_DIR = PATIENT_DIR / "segmentation_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MASK_EXTENSIONS = [".png", ".tif", ".tiff"]

# Object matching criterion:
# A predicted object is considered a true-positive object if
# best IoU with any manual object is >= this threshold.
OBJECT_IOU_THRESHOLD = 0.10


# =========================
# HELPERS
# =========================

def natural_key(path):
    return [
        int(t) if t.isdigit() else t.lower()
        for t in re.split(r"(\d+)", str(path.name))
    ]


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


def read_mask(path):
    if path.suffix.lower() in [".tif", ".tiff"]:
        arr = tiff.imread(path)
    else:
        arr = np.array(Image.open(path))

    if arr.ndim == 3:
        arr = arr[..., 0]

    return arr > 0


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


def collect_mask_files(folder):
    if not folder.exists():
        return []

    files = []
    for ext in MASK_EXTENSIONS:
        files.extend(folder.glob(f"*{ext}"))

    return sorted(files, key=natural_key)


def match_by_tile(tile_number, files):
    for f in files:
        if extract_tile_number(f.name) == tile_number:
            return f
    return None


# =========================
# METRICS
# =========================

def pixel_level_metrics(gt, pred):
    """
    Pixel-level metrics.

    dice:
        2TP / (2TP + FP + FN)

    iou:
        TP / (TP + FP + FN)

    precision:
        TP / (TP + FP)
    """
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    tp = np.logical_and(gt, pred).sum()
    fp = np.logical_and(~gt, pred).sum()
    fn = np.logical_and(gt, ~pred).sum()

    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else np.nan
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "tp_px": int(tp),
        "fp_px": int(fp),
        "fn_px": int(fn),
    }


def object_level_metrics(gt, pred, iou_threshold=0.30):
    """
    Object-level metrics.

    A predicted object is considered TP if its best IoU with any manual
    GT object is >= iou_threshold.

    object_precision:
        TP_objects / (TP_objects + FP_objects)

    fp_objects:
        Predicted objects not matching any manual cell object.

    relative_cell_count_error:
        abs(N_pred - N_gt) / N_gt
    """
    gt_lab = label(gt, connectivity=2)
    pred_lab = label(pred, connectivity=2)

    gt_ids = np.unique(gt_lab)
    pred_ids = np.unique(pred_lab)

    gt_ids = gt_ids[gt_ids != 0]
    pred_ids = pred_ids[pred_ids != 0]

    n_gt = len(gt_ids)
    n_pred = len(pred_ids)

    matched_gt = set()
    matched_pred = set()

    object_rows = []

    for pred_id in pred_ids:
        pred_obj = pred_lab == pred_id
        pred_area = int(pred_obj.sum())

        best_iou = 0.0
        best_gt_id = None

        overlapping_gt_ids = np.unique(gt_lab[pred_obj])
        overlapping_gt_ids = overlapping_gt_ids[overlapping_gt_ids != 0]

        for gt_id in overlapping_gt_ids:
            gt_obj = gt_lab == gt_id

            intersection = np.logical_and(pred_obj, gt_obj).sum()
            union = np.logical_or(pred_obj, gt_obj).sum()

            iou = intersection / union if union > 0 else 0.0

            if iou > best_iou:
                best_iou = float(iou)
                best_gt_id = int(gt_id)

        is_tp = best_iou >= iou_threshold

        if is_tp:
            matched_pred.add(int(pred_id))
            matched_gt.add(best_gt_id)

        object_rows.append(
            {
                "pred_object_id": int(pred_id),
                "best_gt_object_id": best_gt_id,
                "best_iou": float(best_iou),
                "pred_area_px": pred_area,
                "object_class": "TP" if is_tp else "FP",
            }
        )

    tp_objects = len(matched_pred)
    fp_objects = n_pred - tp_objects
    fn_objects = n_gt - len(matched_gt)

    object_precision = (
        tp_objects / (tp_objects + fp_objects)
        if (tp_objects + fp_objects) > 0
        else np.nan
    )

    cell_count_error = abs(n_pred - n_gt)
    relative_cell_count_error = cell_count_error / n_gt if n_gt > 0 else np.nan

    return {
        "n_gt_objects": int(n_gt),
        "n_pred_objects": int(n_pred),
        "tp_objects": int(tp_objects),
        "fp_objects": int(fp_objects),
        "fn_objects": int(fn_objects),
        "object_precision": float(object_precision),
        "cell_count_error": int(cell_count_error),
        "relative_cell_count_error": float(relative_cell_count_error),
        "object_rows": object_rows,
    }


def evaluate_single_prediction(gt, pred, method_name, meta):
    pred = resize_mask_if_needed(pred, gt.shape)

    pix = pixel_level_metrics(gt, pred)
    obj = object_level_metrics(gt, pred, OBJECT_IOU_THRESHOLD)

    row = {
        **meta,
        "method": method_name,
        "object_iou_threshold": OBJECT_IOU_THRESHOLD,
        **pix,
        **{k: v for k, v in obj.items() if k != "object_rows"},
    }

    object_rows = []
    for obj_row in obj["object_rows"]:
        object_rows.append(
            {
                **meta,
                "method": method_name,
                "object_iou_threshold": OBJECT_IOU_THRESHOLD,
                **obj_row,
            }
        )

    return row, object_rows


# =========================
# MAIN
# =========================

def main():
    tile_metric_rows = []
    object_detail_rows = []

    visit_dirs = sorted(
        [p for p in PATIENT_DIR.glob("visit*") if p.is_dir()],
        key=natural_key,
    )

    for visit_dir in visit_dirs:
        visit_name = visit_dir.name

        mosaic_dirs = sorted(
            [p for p in visit_dir.glob("Mosaic*") if p.is_dir()],
            key=natural_key,
        )

        for mosaic_dir in mosaic_dirs:
            mosaic_name = mosaic_dir.name

            manual_dir = mosaic_dir / MANUAL_MASK_SUBDIR
            peter_dir = mosaic_dir / PETER_MASK_SUBDIR
            final_dir = mosaic_dir / FINAL_MASK_SUBDIR

            manual_files = collect_mask_files(manual_dir)

            if len(manual_files) == 0:
                continue

            peter_files = collect_mask_files(peter_dir)
            final_files = collect_mask_files(final_dir)

            print(f"\n[PROCESS] {visit_name} | {mosaic_name}")
            print(f"Manual masks: {len(manual_files)}")

            for manual_file in manual_files:
                tile_number = extract_tile_number(manual_file.name)

                if tile_number is None:
                    print(f"[WARNING] Could not extract tile number: {manual_file}")
                    continue

                peter_file = match_by_tile(tile_number, peter_files)
                final_file = match_by_tile(tile_number, final_files)

                gt = read_mask(manual_file)

                meta = {
                    "visit": visit_name,
                    "mosaic": mosaic_name,
                    "tile": tile_number,
                    "manual_mask": str(manual_file),
                    "peter_mask": str(peter_file) if peter_file else "",
                    "final_mask": str(final_file) if final_file else "",
                }

                if peter_file is not None:
                    peter_pred = read_mask(peter_file)

                    row, obj_rows = evaluate_single_prediction(
                        gt=gt,
                        pred=peter_pred,
                        method_name="PeterChang_UNet",
                        meta=meta,
                    )

                    tile_metric_rows.append(row)
                    object_detail_rows.extend(obj_rows)

                else:
                    print(f"[WARNING] Missing Peter Chang mask for tile {tile_number}")

                if final_file is not None:
                    final_pred = read_mask(final_file)

                    row, obj_rows = evaluate_single_prediction(
                        gt=gt,
                        pred=final_pred,
                        method_name="PeterChang_area_FLIM",
                        meta=meta,
                    )

                    tile_metric_rows.append(row)
                    object_detail_rows.extend(obj_rows)

                else:
                    print(f"[WARNING] Missing final mask for tile {tile_number}")

    if len(tile_metric_rows) == 0:
        print("\nNo evaluation rows generated.")
        return

    df_tiles = pd.DataFrame(tile_metric_rows)
    df_objects = pd.DataFrame(object_detail_rows)

    metric_cols = [
        "dice",
        "iou",
        "precision",
        "object_precision",
        "fp_objects",
        "cell_count_error",
        "relative_cell_count_error",
    ]

    tile_csv = OUTPUT_DIR / "segmentation_key_metrics_by_tile.csv"
    object_csv = OUTPUT_DIR / "segmentation_object_matching_details.csv"
    summary_csv = OUTPUT_DIR / "segmentation_key_metrics_summary.csv"
    summary_by_visit_csv = OUTPUT_DIR / "segmentation_key_metrics_summary_by_visit.csv"

    df_tiles.to_csv(tile_csv, index=False)
    df_objects.to_csv(object_csv, index=False)

    summary = (
        df_tiles
        .groupby("method")[metric_cols]
        .agg(["mean", "std", "median", "count"])
        .reset_index()
    )

    summary_by_visit = (
        df_tiles
        .groupby(["visit", "method"])[metric_cols]
        .agg(["mean", "std", "median", "count"])
        .reset_index()
    )

    summary.to_csv(summary_csv, index=False)
    summary_by_visit.to_csv(summary_by_visit_csv, index=False)

    print("\n[DONE]")
    print(f"Tile metrics:\n{tile_csv}")
    print(f"Object matching details:\n{object_csv}")
    print(f"Summary:\n{summary_csv}")
    print(f"Summary by visit:\n{summary_by_visit_csv}")


if __name__ == "__main__":
    main()