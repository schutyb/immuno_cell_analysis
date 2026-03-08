#!/usr/bin/env python3

import argparse
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from matplotlib.colors import LinearSegmentedColormap
from phasorpy.plot import PhasorPlot


FREQUENCY_MHZ = 80.0
DPI = 400

# Separate binning factors
PHASOR_BINNING = 8
PSEUDOCOLOR_BINNING = 4


# ----------------------------
# IO
# ----------------------------
def read_tiff(path: Path) -> np.ndarray:
    arr = tiff.imread(str(path))
    return np.asarray(arr)


def save_rgb_png(path: Path, img: np.ndarray) -> None:
    arr = np.asarray(img, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    iio.imwrite(str(path), (255 * arr).astype(np.uint8))


# ----------------------------
# Utils
# ----------------------------
def normalize_percentile(x: np.ndarray, pmin: float = 2.0, pmax: float = 98.0) -> np.ndarray:
    """
    Normalize using percentiles, ignoring NaNs/Infs.
    Invalid pixels are set to 0 in the output.
    """
    arr = np.asarray(x, dtype=np.float32)
    good = np.isfinite(arr)

    out = np.zeros_like(arr, dtype=np.float32)
    if not np.any(good):
        return out

    lo = np.percentile(arr[good], pmin)
    hi = np.percentile(arr[good], pmax)

    if hi <= lo:
        return out

    out[good] = (arr[good] - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    out[~good] = 0.0
    return out


def block_mean_2d(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return arr

    h, w = arr.shape
    h2 = (h // factor) * factor
    w2 = (w // factor) * factor

    arr = arr[:h2, :w2]
    return arr.reshape(h2 // factor, factor, w2 // factor, factor).mean(axis=(1, 3))


def upsample_nearest(arr: np.ndarray, factor: int, out_shape: tuple[int, int]) -> np.ndarray:
    """
    Upsample 2D or 3D image by nearest-neighbor repetition, then crop.
    """
    if factor <= 1:
        return arr

    if arr.ndim == 2:
        up = np.repeat(np.repeat(arr, factor, axis=0), factor, axis=1)
        return up[:out_shape[0], :out_shape[1]]

    if arr.ndim == 3:
        up = np.repeat(np.repeat(arr, factor, axis=0), factor, axis=1)
        return up[:out_shape[0], :out_shape[1], :]

    raise ValueError(f"Unsupported ndim={arr.ndim}")


def make_phase_colormap() -> LinearSegmentedColormap:
    colors = [
        (0.00, "#6A00FF"),  # violeta
        (0.10, "#3A00FF"),  # violeta-azul
        (0.20, "#304FFE"),  # azul
        (0.32, "#0080FF"),  # azul claro
        (0.45, "#00B050"),  # verde
        (0.55, "#8FD400"),  # verde-amarillo
        (0.65, "#FFF200"),  # amarillo
        (0.75, "#FFC000"),  # amarillo-naranja
        (0.85, "#FF9E00"),  # naranja
        (0.92, "#FF5500"),  # naranja-rojo
        (1.00, "#E60000"),  # rojo
    ]
    return LinearSegmentedColormap.from_list("phase_map", colors, N=256)


def make_black_alpha_cmap():
    return LinearSegmentedColormap.from_list(
        "black_alpha",
        [
            (1.0, 1.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 0.30),
            (0.0, 0.0, 0.0, 0.55),
            (0.0, 0.0, 0.0, 0.80),
            (0.0, 0.0, 0.0, 1.00),
        ],
        N=256,
    )


def phase_intensity_to_rgb(g: np.ndarray, s: np.ndarray, avg: np.ndarray) -> np.ndarray:
    """
    hue = phase
    brightness = intensity

    NaN-safe:
    - phase is computed only on valid pixels
    - invalid pixels become black

    IMPORTANT:
    This keeps EXACTLY the same pseudocolor mapping you had before.
    """
    g = np.asarray(g, dtype=np.float32)
    s = np.asarray(s, dtype=np.float32)
    avg = np.asarray(avg, dtype=np.float32)

    valid = np.isfinite(g) & np.isfinite(s) & np.isfinite(avg) & (avg > 0)

    phase = np.full_like(g, np.nan, dtype=np.float32)
    phase[valid] = np.arctan2(s[valid], g[valid])

    # Keep pseudocolor exactly as before
    phase_norm = normalize_percentile(phase, pmin=1.0, pmax=99.0)
    phase_norm = np.power(phase_norm, 0.3)

    intensity_norm = normalize_percentile(avg, pmin=1.0, pmax=99.0)
    intensity_norm = intensity_norm ** 0.99

    cmap = make_phase_colormap()
    rgb = cmap(phase_norm)[..., :3]
    rgb *= intensity_norm[..., None]

    rgb[~valid] = 0.0
    return np.clip(rgb, 0.0, 1.0)


def add_phasor_rainbow_background(
    ax,
    real_data: np.ndarray,
    imag_data: np.ndarray,
    alpha: float = 0.50,
    nx: int = 700,
    ny: int = 450,
) -> None:
    """
    Add a phase-colored rainbow background under the universal semicircle,
    leaving the rest white, with a slightly warmer appearance than before.

    This affects ONLY the phasor background, not the pseudocolor image.
    """
    g = np.linspace(0.0, 1.0, nx)
    s = np.linspace(0.0, 0.65, ny)
    gg, ss = np.meshgrid(g, s)

    # Universal semicircle
    inside_semicircle = ((gg - 0.5) ** 2 + ss ** 2 <= 0.25) & (ss >= 0)

    # Phase in phasor space
    phase_bg = np.arctan2(ss, gg)

    # Radius from origin
    radius = np.sqrt(gg**2 + ss**2)

    # Actual phase max from plotted data
    valid_data = np.isfinite(real_data) & np.isfinite(imag_data)
    if np.any(valid_data):
        phase_data = np.arctan2(imag_data[valid_data], real_data[valid_data])
        phase_max = float(np.nanmax(phase_data))
        if phase_max <= 0:
            phase_max = 1.0
    else:
        phase_max = 1.0

    phase_min = 0.0
    phase_norm = (phase_bg - phase_min) / (phase_max - phase_min)
    phase_norm = np.clip(phase_norm, 0.0, 1.0)

    # Slightly less aggressive cool-color expansion than before:
    # warmer background without touching the pseudocolor image.
    phase_norm = np.power(phase_norm, 0.55)

    cmap = make_phase_colormap()
    rgb_inside = cmap(phase_norm)[..., :3]

    # More arc-like appearance:
    # whiter near the origin, stronger color near the semicircle
    semicircle_radius = 0.5
    whiten = np.clip(radius / semicircle_radius, 0.0, 1.0)
    whiten = np.power(whiten, 1.8)

    rgb_mixed = (1.0 - whiten[..., None]) * np.ones_like(rgb_inside) + whiten[..., None] * rgb_inside

    # White outside the semicircle
    rgb = np.ones((ny, nx, 3), dtype=np.float32)
    rgb[inside_semicircle] = rgb_mixed[inside_semicircle]

    ax.imshow(
        rgb,
        origin="lower",
        extent=[0.0, 1.0, 0.0, 0.65],
        aspect="auto",
        alpha=alpha,
        zorder=0,
    )


# ----------------------------
# Plotting
# ----------------------------
def save_phasor_plot_png(out_path: Path, g: np.ndarray, s: np.ndarray, avg: np.ndarray) -> None:
    # 8x8 binning before plotting
    g_bin = block_mean_2d(g, PHASOR_BINNING)
    s_bin = block_mean_2d(s, PHASOR_BINNING)
    avg_bin = block_mean_2d(avg, PHASOR_BINNING)

    good = np.isfinite(g_bin) & np.isfinite(s_bin) & np.isfinite(avg_bin)

    if np.any(good):
        thr = np.percentile(avg_bin[good], 5)
        good = good & (avg_bin > thr)

    real = g_bin[good]
    imag = s_bin[good]

    if real.size == 0:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.text(0.5, 0.5, "No valid phasor pixels", ha="center", va="center", fontsize=14)
        ax.axis("off")
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        return

    plot = PhasorPlot(frequency=FREQUENCY_MHZ, title="First harmonic phasor plot")
    ax = plot.ax

    # Rainbow background first
    add_phasor_rainbow_background(ax, real, imag, alpha=0.50)

    # Histogram on top with transparent-to-black cmap
    plot.hist2d(real, imag, cmap=make_black_alpha_cmap(), bins=128)

    # Redraw semicircle on top
    plot.semicircle()

    fig = plt.gcf()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def save_pseudocolor_pngs(base_path_no_ext: Path, g: np.ndarray, s: np.ndarray, avg: np.ndarray) -> None:
    """
    Saves:
      - binned pseudocolor
      - upsampled pseudocolor back to original size
    """
    g_bin = block_mean_2d(g, PSEUDOCOLOR_BINNING)
    s_bin = block_mean_2d(s, PSEUDOCOLOR_BINNING)
    avg_bin = block_mean_2d(avg, PSEUDOCOLOR_BINNING)

    rgb_binned = phase_intensity_to_rgb(g_bin, s_bin, avg_bin)
    save_rgb_png(base_path_no_ext.parent / f"{base_path_no_ext.name}_binned.png", rgb_binned)

    rgb_upsampled = upsample_nearest(rgb_binned, PSEUDOCOLOR_BINNING, g.shape)
    save_rgb_png(base_path_no_ext.parent / f"{base_path_no_ext.name}_upsampled.png", rgb_upsampled)


# ----------------------------
# Processing
# ----------------------------
def process_new_dir(new_dir: Path) -> None:
    phasor_path = new_dir / "phasor.tif"
    if not phasor_path.exists():
        print(f"[SKIP] No phasor.tif in {new_dir}")
        return

    stack = read_tiff(phasor_path)

    if stack.ndim < 3 or stack.shape[0] < 3:
        print(f"[SKIP] Unexpected phasor stack shape in {phasor_path}: {stack.shape}")
        return

    avg = stack[0].astype(np.float32)
    g1 = stack[1].astype(np.float32)
    s1 = stack[2].astype(np.float32)

    save_phasor_plot_png(new_dir / "phasor_first_harmonic.png", g1, s1, avg)
    save_pseudocolor_pngs(new_dir / "pseudocolor_phase_first_harmonic", g1, s1, avg)

    print(f"[OK] Saved in {new_dir}")


def find_new_dirs(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("_new") if p.is_dir()])


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    DEFAULT_PATIENT_DIR = Path(
        "/Users/schutyb/Documents/balu_lab/dod/data_raw/patients/p449"
    ).expanduser()

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--patient-dir",
        type=str,
        default=str(DEFAULT_PATIENT_DIR),
        help="Carpeta general del paciente donde están todas las visitas",
    )
    args = ap.parse_args()

    patient_dir = Path(args.patient_dir).expanduser().resolve()
    if not patient_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta: {patient_dir}")

    new_dirs = find_new_dirs(patient_dir)
    if not new_dirs:
        raise RuntimeError(f"No se encontraron carpetas _new dentro de {patient_dir}")

    print(f"[INFO] _new folders found: {len(new_dirs)}")
    for new_dir in new_dirs:
        process_new_dir(new_dir)

    print("[DONE] Finished.")


if __name__ == "__main__":
    main()