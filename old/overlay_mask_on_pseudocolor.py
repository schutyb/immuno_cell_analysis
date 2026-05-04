#!/usr/bin/env python3

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector


# ----------------------------
# CONFIG
# ----------------------------

color_image_path = Path(
    "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449/visit01/Mosaic03_4x4_FOV600_z110_32Sp/_new/pseudocolor_phase_first_harmonic_upsampled.png"
).expanduser()

mask_image_path = Path(
    "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449/visit01/Mosaic03_4x4_FOV600_z110_32Sp/_new/instance_mask_filtered.png"
).expanduser()
# también puede ser .tif si querés

output_dir = color_image_path.parent / "roi_overlay_from_filtered_mask"


# ----------------------------
# IO
# ----------------------------

def read_image(path):
    if path.suffix.lower() in [".tif", ".tiff"]:
        arr = tiff.imread(str(path))
    else:
        arr = iio.imread(str(path))

    arr = np.asarray(arr)

    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]

    return arr


def read_rgb_image(path):
    arr = read_image(path).astype(np.float32)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Invalid RGB image shape: {arr.shape}")

    if arr.max() > 1:
        arr = arr / 255.0

    return np.clip(arr, 0, 1)


def read_mask_image(path):
    arr = read_image(path)

    if arr.ndim == 3:
        arr = arr[..., 0]

    if arr.ndim != 2:
        raise ValueError(f"Invalid mask image shape: {arr.shape}")

    return arr


def save_png(path, img):
    img = np.asarray(img)

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    img = np.clip(img, 0, 1)
    iio.imwrite(str(path), (255 * img).astype(np.uint8))


def save_binary_mask_png(path, mask):
    mask = (mask > 0).astype(np.uint8)
    rgb = np.stack([mask, mask, mask], axis=-1).astype(np.float32)
    save_png(path, rgb)


def save_gif(path, frames, duration_ms=700, loop=0):
    frames_uint8 = []
    for f in frames:
        f = np.asarray(f)
        if f.ndim == 2:
            f = np.stack([f, f, f], axis=-1)
        f = np.clip(f, 0, 1)
        frames_uint8.append((255 * f).astype(np.uint8))

    iio.imwrite(str(path), frames_uint8, duration=duration_ms / 1000.0, loop=loop)


# ----------------------------
# ROI SELECTION
# ----------------------------

def select_roi_interactive(rgb_image):
    """
    Selección manual rectangular.
    Dibujás la ROI con el mouse y cerrás la ventana para continuar.
    Esc cancela.
    """
    roi = {"coords": None, "cancelled": False}

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb_image)
    ax.set_title(
        "Dibujá una ROI rectangular con el mouse\n"
        "Cerrá la ventana para continuar | Esc = cancelar"
    )
    ax.axis("on")

    def on_select(eclick, erelease):
        x0, y0 = eclick.xdata, eclick.ydata
        x1, y1 = erelease.xdata, erelease.ydata

        if None in [x0, y0, x1, y1]:
            return

        x0, x1 = sorted([int(round(x0)), int(round(x1))])
        y0, y1 = sorted([int(round(y0)), int(round(y1))])

        roi["coords"] = (x0, y0, x1, y1)
        print(f"[INFO] ROI seleccionada: x={x0}:{x1}, y={y0}:{y1}")

    RectangleSelector(
        ax,
        on_select,
        useblit=True,
        button=[1],
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=True,
    )

    def on_key(event):
        if event.key == "escape":
            roi["cancelled"] = True
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    if roi["cancelled"]:
        raise RuntimeError("Selección de ROI cancelada por el usuario.")

    if roi["coords"] is None:
        raise RuntimeError("No se seleccionó ninguna ROI.")

    x0, y0, x1, y1 = roi["coords"]

    h, w = rgb_image.shape[:2]
    x0 = max(0, min(x0, w - 1))
    x1 = max(0, min(x1, w))
    y0 = max(0, min(y0, h - 1))
    y1 = max(0, min(y1, h))

    if x1 <= x0 or y1 <= y0:
        raise RuntimeError("ROI inválida.")

    return x0, y0, x1, y1


# ----------------------------
# OVERLAY
# ----------------------------

def overlay_mask(rgb, mask, overlay_color=(1.0, 0.0, 0.0)):
    """
    Overlay de toda la máscara sobre toda la imagen.
    """
    out = rgb.copy()
    mask_bool = mask > 0
    out[mask_bool] = overlay_color
    return out


def crop_roi(arr, roi):
    x0, y0, x1, y1 = roi
    if arr.ndim == 2:
        return arr[y0:y1, x0:x1]
    elif arr.ndim == 3:
        return arr[y0:y1, x0:x1, :]
    else:
        raise ValueError(f"Unsupported array shape for crop: {arr.shape}")


# ----------------------------
# MAIN
# ----------------------------

def main():
    output_dir.mkdir(parents=True, exist_ok=True)

    # leer pseudocolor y máscara filtrada
    pseudo = read_rgb_image(color_image_path)
    mask = read_mask_image(mask_image_path)
    mask = (mask > 0).astype(np.uint8)

    if pseudo.shape[:2] != mask.shape:
        raise ValueError(
            f"Size mismatch:\n"
            f"  pseudocolor = {pseudo.shape[:2]}\n"
            f"  mask       = {mask.shape}"
        )

    # overlay completo en rojo
    overlay_full = overlay_mask(pseudo, mask, overlay_color=(1.0, 0.0, 0.0))

    # guardar outputs completos
    save_png(output_dir / "pseudocolor_full.png", pseudo)
    save_binary_mask_png(output_dir / "mask_full_binary.png", mask)
    save_png(output_dir / "pseudocolor_overlay_full_red.png", overlay_full)

    save_gif(
        output_dir / "pseudocolor_overlay_full_red.gif",
        [pseudo, overlay_full],
        duration_ms=700,
        loop=0,
    )

    print("[OK] Guardado overlay completo y GIF completo.")

    # seleccionar ROI manual sobre pseudocolor
    roi = select_roi_interactive(pseudo)
    x0, y0, x1, y1 = roi
    print(f"[OK] ROI final: x={x0}:{x1}, y={y0}:{y1}")

    # crops
    pseudo_roi = crop_roi(pseudo, roi)
    mask_roi = crop_roi(mask, roi)
    overlay_roi = overlay_mask(pseudo_roi, mask_roi, overlay_color=(1.0, 0.0, 0.0))

    # guardar outputs ROI
    save_png(output_dir / "selected_roi_pseudocolor.png", pseudo_roi)
    save_binary_mask_png(output_dir / "selected_roi_mask_binary.png", mask_roi)
    save_png(output_dir / "selected_roi_overlay_red.png", overlay_roi)

    save_gif(
        output_dir / "selected_roi_overlay_red.gif",
        [pseudo_roi, overlay_roi],
        duration_ms=700,
        loop=0,
    )

    # guardar coordenadas ROI
    roi_txt = output_dir / "selected_roi_coordinates.txt"
    with open(roi_txt, "w") as f:
        f.write(f"x0={x0}\n")
        f.write(f"y0={y0}\n")
        f.write(f"x1={x1}\n")
        f.write(f"y1={y1}\n")
        f.write(f"width={x1 - x0}\n")
        f.write(f"height={y1 - y0}\n")

    print("[OK] Guardado ROI recortada, máscara ROI, overlay ROI y GIF ROI.")
    print("[OK] Resultados en:", output_dir)


if __name__ == "__main__":
    main()