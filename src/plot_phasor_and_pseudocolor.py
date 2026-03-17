#!/usr/bin/env python3

import argparse
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from matplotlib.colors import LinearSegmentedColormap
from phasorpy.plot import PhasorPlot

from color_scales import (
    get_phase_colormap,
    map_phase_deg_to_norm,
    phase_intensity_to_rgb,
    phase_rad_to_deg,
)


FREQUENCY_MHZ = 80.0
DPI = 400

# Separate binning factors
PHASOR_BINNING = 8
PSEUDOCOLOR_BINNING = 4

# Manual phase range in DEGREES
PHASE_MIN_DEG = 0.0
PHASE_MAX_DEG = 55.0

# Optional gamma on phase mapping
PHASE_GAMMA = 0.6

# Color scale from color_scales.py
PHASE_SCALE = "reds_to_greens"
PHASE_CMAP_LEVELS = 2048


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
    if factor <= 1:
        return arr

    if arr.ndim == 2:
        up = np.repeat(np.repeat(arr, factor, axis=0), factor, axis=1)
        return up[:out_shape[0], :out_shape[1]]

    if arr.ndim == 3:
        up = np.repeat(np.repeat(arr, factor, axis=0), factor, axis=1)
        return up[:out_shape[0], :out_shape[1], :]

    raise ValueError(f"Unsupported ndim={arr.ndim}")


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


def add_phasor_rainbow_background(
    ax,
    alpha: float = 0.50,
    nx: int = 700,
    ny: int = 450,
) -> None:
    """
    Phasor background using the selected scale from color_scales.py.

    Only colors the region where:
        PHASE_MIN_DEG <= phase <= PHASE_MAX_DEG

    Everything else stays white.
    """
    g = np.linspace(0.0, 1.0, nx)
    s = np.linspace(0.0, 0.65, ny)
    gg, ss = np.meshgrid(g, s)

    # Universal semicircle
    inside_semicircle = ((gg - 0.5) ** 2 + ss ** 2 <= 0.25) & (ss >= 0)

    # Phase in phasor space
    phase_bg_rad = np.arctan2(ss, gg)
    phase_bg_deg = phase_rad_to_deg(phase_bg_rad)

    # Keep only desired phase range
    inside_phase_range = (
        np.isfinite(phase_bg_deg)
        & (phase_bg_deg >= PHASE_MIN_DEG)
        & (phase_bg_deg <= PHASE_MAX_DEG)
    )

    valid_bg = inside_semicircle & inside_phase_range

    # Start with white everywhere
    rgb = np.ones((ny, nx, 3), dtype=np.float32)

    if np.any(valid_bg):
        phase_norm = np.zeros_like(phase_bg_deg, dtype=np.float32)
        phase_norm[valid_bg] = map_phase_deg_to_norm(
            phase_bg_deg[valid_bg],
            phase_min_deg=PHASE_MIN_DEG,
            phase_max_deg=PHASE_MAX_DEG,
            phase_gamma=PHASE_GAMMA,
        )

        cmap = get_phase_colormap(PHASE_SCALE, n=PHASE_CMAP_LEVELS)
        rgb_inside = cmap(phase_norm)[..., :3]

        # Arc-like effect: whiter near origin, stronger near semicircle
        radius = np.sqrt(gg**2 + ss**2)
        semicircle_radius = 0.5
        whiten = np.clip(radius / semicircle_radius, 0.0, 1.0)
        whiten = np.power(whiten, 1.8)

        rgb_mixed = np.ones_like(rgb_inside)
        rgb_mixed[valid_bg] = (
            (1.0 - whiten[valid_bg, None]) * np.ones((np.sum(valid_bg), 3), dtype=np.float32)
            + whiten[valid_bg, None] * rgb_inside[valid_bg]
        )

        rgb[valid_bg] = rgb_mixed[valid_bg]

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

    add_phasor_rainbow_background(ax, alpha=0.50)
    plot.hist2d(real, imag, cmap=make_black_alpha_cmap(), bins=128)
    plot.semicircle()

    fig = plt.gcf()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def save_pseudocolor_pngs(base_path_no_ext: Path, g: np.ndarray, s: np.ndarray, avg: np.ndarray) -> None:
    g_bin = block_mean_2d(g, PSEUDOCOLOR_BINNING)
    s_bin = block_mean_2d(s, PSEUDOCOLOR_BINNING)
    avg_bin = block_mean_2d(avg, PSEUDOCOLOR_BINNING)

    rgb_binned = phase_intensity_to_rgb(
        g_bin,
        s_bin,
        avg_bin,
        scale=PHASE_SCALE,
        phase_min_deg=PHASE_MIN_DEG,
        phase_max_deg=PHASE_MAX_DEG,
        phase_gamma=PHASE_GAMMA,
        n=PHASE_CMAP_LEVELS,
    )
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

    print(f"[INFO] Phase range (deg): {PHASE_MIN_DEG} -> {PHASE_MAX_DEG}")
    print(f"[INFO] Phase scale: {PHASE_SCALE}")
    print(f"[INFO] _new folders found: {len(new_dirs)}")

    for new_dir in new_dirs:
        process_new_dir(new_dir)

    print("[DONE] Finished.")


if __name__ == "__main__":
    main()