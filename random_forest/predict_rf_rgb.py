from pathlib import Path

import joblib
import numpy as np
import tifffile as tiff
from PIL import Image, ImageDraw, ImageFont

from skimage.color import rgb2hsv, rgb2lab, rgb2gray
from skimage.filters import gaussian, sobel, laplace
from skimage.morphology import remove_small_objects, remove_small_holes


MODEL_PATH = "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449/models/rf_rgb_visit01.joblib"

INPUT_FOLDER = "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449"

PROB_THRESHOLD = 0.50
MIN_OBJECT_SIZE = 20
MAX_HOLE_SIZE = 20


def load_rgb(path):
    img = tiff.imread(path)
    return img.astype(np.uint8)


def extract_rgb_features(img):
    img_f = img.astype(np.float32) / 255.0

    R = img_f[..., 0]
    G = img_f[..., 1]
    B = img_f[..., 2]

    total = R + G + B + 1e-8

    r_norm = R / total
    g_norm = G / total
    b_norm = B / total

    hsv = rgb2hsv(img_f)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    lab = rgb2lab(img_f)
    L = lab[..., 0] / 100.0
    A = lab[..., 1] / 128.0
    BB = lab[..., 2] / 128.0

    gray = rgb2gray(img_f).astype(np.float32)
    grad = sobel(gray).astype(np.float32)

    gauss1 = gaussian(gray, sigma=1, preserve_range=True).astype(np.float32)
    gauss2 = gaussian(gray, sigma=2, preserve_range=True).astype(np.float32)
    gauss4 = gaussian(gray, sigma=4, preserve_range=True).astype(np.float32)

    log1 = laplace(gauss1).astype(np.float32)
    log2 = laplace(gauss2).astype(np.float32)

    features = np.stack(
        [
            R, G, B,
            r_norm, g_norm, b_norm,
            R - G, R - B, G - B,
            R / (G + 1e-8),
            R / (B + 1e-8),
            G / (B + 1e-8),
            H, S, V,
            L, A, BB,
            gray,
            grad,
            gauss1, gauss2, gauss4,
            log1, log2,
        ],
        axis=-1,
    )

    return features.reshape(-1, features.shape[-1]).astype(np.float32)


def predict_tile(clf, img):
    X = extract_rgb_features(img)
    prob = clf.predict_proba(X)[:, 1]
    prob = prob.reshape(img.shape[:2]).astype(np.float32)

    mask = prob > PROB_THRESHOLD
    mask = remove_small_objects(mask, min_size=MIN_OBJECT_SIZE)
    mask = remove_small_holes(mask, area_threshold=MAX_HOLE_SIZE)

    return prob, mask.astype(np.uint8)


def make_red_overlay(img, mask, alpha=0.70):
    overlay = img.copy().astype(np.float32)

    red = np.zeros_like(overlay)
    red[..., 0] = 255

    overlay[mask > 0] = (
        overlay[mask > 0] * (1 - alpha)
        + red[mask > 0] * alpha
    )

    return np.clip(overlay, 0, 255).astype(np.uint8)


def mask_to_rgb(mask):
    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    mask_rgb[mask > 0] = [255, 255, 255]
    return mask_rgb


def make_panel(rgb, mask, overlay, title):
    h, w, _ = rgb.shape

    title_h = 80
    gap = 10

    panel_w = w * 3 + gap * 2
    panel_h = h + title_h

    canvas = Image.new("RGB", (panel_w, panel_h), "white")

    rgb_img = Image.fromarray(rgb)
    mask_img = Image.fromarray(mask_to_rgb(mask))
    overlay_img = Image.fromarray(overlay)

    canvas.paste(rgb_img, (0, title_h))
    canvas.paste(mask_img, (w + gap, title_h))
    canvas.paste(overlay_img, (2 * w + 2 * gap, title_h))

    draw = ImageDraw.Draw(canvas)

    try:
        font_title = ImageFont.truetype("Arial.ttf", 28)
        font_label = ImageFont.truetype("Arial.ttf", 24)
    except:
        font_title = None
        font_label = None

    draw.text((10, 8), title, fill="black", font=font_title)
    draw.text((10, 45), "RGB", fill="black", font=font_label)
    draw.text((w + gap + 10, 45), "Predicted mask", fill="black", font=font_label)
    draw.text((2 * w + 2 * gap + 10, 45), "Overlay", fill="black", font=font_label)

    return canvas


def find_mosaic_dirs(input_folder):
    input_folder = Path(input_folder)

    if input_folder.name.lower().startswith("visit"):
        return sorted([
            p for p in input_folder.iterdir()
            if p.is_dir() and p.name.lower().startswith("mosaic")
        ])

    return sorted([
        p for p in input_folder.glob("visit*/Mosaic*")
        if p.is_dir()
    ])


def process_mosaic(clf, mosaic_dir):
    rgb_dir = mosaic_dir / "random_forest" / "rgb"
    out_dir = mosaic_dir / "random_forest" / "prediction_rgb_rf_tiles"
    out_dir.mkdir(parents=True, exist_ok=True)

    tile_paths = sorted(rgb_dir.glob("Im_*_pseudoRGB.tif"))

    if len(tile_paths) == 0:
        print(f"Saltando {mosaic_dir.name}: no encontré tiles RGB")
        return

    print(f"\nProcesando mosaico: {mosaic_dir}")
    print(f"Tiles encontrados: {len(tile_paths)}")

    for tile_path in tile_paths:
        img = load_rgb(tile_path)
        prob, mask = predict_tile(clf, img)
        overlay = make_red_overlay(img, mask)

        tile_id = tile_path.stem.replace("_pseudoRGB", "")
        title = f"{mosaic_dir.parent.name} | {mosaic_dir.name} | {tile_id}"

        panel = make_panel(
            rgb=img,
            mask=mask,
            overlay=overlay,
            title=title
        )

        out_path = out_dir / f"{tile_id}_rgb_mask_overlay_panel.png"
        panel.save(out_path, dpi=(600, 600))

        print(f"Guardado: {out_path}")


def main():
    clf = joblib.load(MODEL_PATH)

    mosaic_dirs = find_mosaic_dirs(INPUT_FOLDER)

    print(f"Modelo cargado: {MODEL_PATH}")
    print(f"Mosaicos encontrados: {len(mosaic_dirs)}")

    for mosaic_dir in mosaic_dirs:
        process_mosaic(clf, mosaic_dir)


if __name__ == "__main__":
    main()