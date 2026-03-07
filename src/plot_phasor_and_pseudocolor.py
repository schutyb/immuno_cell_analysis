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
    arr = np.asarray(x, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    good = np.isfinite(arr)
    if not np.any(good):
        return np.zeros_like(arr)

    lo = np.percentile(arr[good], pmin)
    hi = np.percentile(arr[good], pmax)

    if hi <= lo:
        return np.zeros_like(arr)

    y = (arr - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0)


def block_mean_2d(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return arr

    h, w = arr.shape
    h2 = (h // factor) * factor
    w2 = (w // factor) * factor

    arr = arr[:h2, :w2]
    return arr.reshape(h2 // factor, factor, w2 // factor, factor).mean(axis=(1, 3))


def make_phase_colormap() -> LinearSegmentedColormap:
    colors = [
        (0.00, "#6A00FF"),  # violeta
        (0.18, "#304FFE"),  # azul
        (0.40, "#00B050"),  # verde
        (0.62, "#FFF200"),  # amarillo
        (0.80, "#FF9E00"),  # naranja
        (1.00, "#E60000"),  # rojo
    ]
    return LinearSegmentedColormap.from_list("phase_map", colors, N=256)


def phase_intensity_to_rgb(g: np.ndarray, s: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    """
    hue = phase
    brightness = intensity
    """
    phase = np.arctan2(s, g)

    phase_norm = normalize_percentile(phase, pmin=2.0, pmax=98.0)
    phase_norm = np.power(phase_norm, 0.85)

    intensity_norm = normalize_percentile(intensity, pmin=1.0, pmax=99.0)
    intensity_norm = intensity_norm ** 0.7

    cmap = make_phase_colormap()
    rgb = cmap(phase_norm)[..., :3]
    rgb *= intensity_norm[..., None]

    return np.clip(rgb, 0.0, 1.0)


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

    plot = PhasorPlot(frequency=FREQUENCY_MHZ, title="First harmonic phasor")
    plot.hist2d(real, imag, cmap="RdYlGn_r", bins=200)

    fig = plt.gcf()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def save_pseudocolor_png(out_path: Path, g: np.ndarray, s: np.ndarray, avg: np.ndarray) -> None:
    # 4x4 binning before pseudocolor
    g_bin = block_mean_2d(g, PSEUDOCOLOR_BINNING)
    s_bin = block_mean_2d(s, PSEUDOCOLOR_BINNING)
    avg_bin = block_mean_2d(avg, PSEUDOCOLOR_BINNING)

    rgb = phase_intensity_to_rgb(g_bin, s_bin, avg_bin)
    save_rgb_png(out_path, rgb)


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
    save_pseudocolor_png(new_dir / "pseudocolor_phase_first_harmonic.png", g1, s1, avg)

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