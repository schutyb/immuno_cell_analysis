#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utilities for FLIM phasor color mapping and pseudocolor image generation.

This module provides reusable functions to:

1. Normalize scalar arrays using robust percentile-based scaling.
2. Convert phasor phase from radians to degrees.
3. Map phase values to normalized color coordinates.
4. Build reusable colormaps for FLIM visualization.
5. Convert phase-only or phasor-plus-intensity data into RGB images.
6. Generate phasor backgrounds and phase legend strips.

The main use case is fluorescence lifetime imaging microscopy (FLIM)
visualization in phasor space, especially for pseudocolor rendering of
phasor-derived phase information from calibrated G and S coordinates.

Three colormap families are currently included:

- ``spectral``:
    Full inverted spectrum, designed for broad phase/lifetime coverage.

- ``reds_to_greens``:
    A red-to-green scale intended for green-detector FLIM visualization.
    It maps lower phase values to pale/red/orange tones and higher phase
    values to green tones.

- ``blues_to_greens``:
    A blue-to-green scale intended for blue-detector FLIM visualization.
    It is complementary to ``reds_to_greens`` and maps lower phase values
    to pale blue / blue / cyan tones and higher phase values to green tones.

This allows two detector channels to be visualized with related but distinct
continuous phase scales:

- green detector: red/orange/yellow -> green
- blue detector: blue/cyan/teal -> green

All RGB outputs are returned as floating-point arrays in the range [0, 1].
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ScaleName = Literal["spectral", "reds_to_greens", "blues_to_greens"]


def normalize_percentile(
    x: np.ndarray,
    pmin: float = 1.0,
    pmax: float = 99.0,
) -> np.ndarray:
    """
    Normalize an array to the range [0, 1] using percentile clipping.

    This function is intended for robust visualization scaling in the presence
    of outliers, noise, or invalid values. Only finite values are used to
    estimate the lower and upper scaling bounds.

    Parameters
    ----------
    x : numpy.ndarray
        Input array to normalize.
    pmin : float, default=1.0
        Lower percentile used as the minimum normalization bound.
    pmax : float, default=99.0
        Upper percentile used as the maximum normalization bound.

    Returns
    -------
    numpy.ndarray
        Array of the same shape as ``x``, normalized to [0, 1].
        Invalid entries are set to 0.

    Notes
    -----
    - If no finite values are present, the function returns an array of zeros.
    - If the estimated upper bound is not greater than the lower bound,
      the function also returns zeros.
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


def phase_rad_to_deg(phase_rad: np.ndarray) -> np.ndarray:
    """
    Convert phase values from radians to degrees.

    Parameters
    ----------
    phase_rad : numpy.ndarray
        Array containing phase values in radians.

    Returns
    -------
    numpy.ndarray
        Array of phase values in degrees, stored as ``float32``.
    """
    return np.degrees(np.asarray(phase_rad, dtype=np.float32)).astype(np.float32)


def map_phase_deg_to_norm(
    phase_deg: np.ndarray,
    phase_min_deg: float,
    phase_max_deg: float,
    phase_gamma: float = 1.0,
) -> np.ndarray:
    """
    Map phase values in degrees to the interval [0, 1].

    This function defines the core phase-to-color coordinate transform.
    It first linearly maps phase values between a user-defined minimum and
    maximum range, then optionally applies a gamma correction to reshape
    the visual contrast.

    Parameters
    ----------
    phase_deg : numpy.ndarray
        Input phase array in degrees.
    phase_min_deg : float
        Lower phase bound mapped to 0.
    phase_max_deg : float
        Upper phase bound mapped to 1.
    phase_gamma : float, default=1.0
        Gamma correction applied after linear mapping.
        Values < 1.0 expand lower values, whereas values > 1.0 compress them.

    Returns
    -------
    numpy.ndarray
        Normalized phase array in the range [0, 1].
        Invalid entries are set to 0.

    Raises
    ------
    ValueError
        If ``phase_max_deg <= phase_min_deg``.

    Notes
    -----
    Values outside the specified phase range are clipped to [0, 1].
    """
    phase_deg = np.asarray(phase_deg, dtype=np.float32)
    out = np.zeros_like(phase_deg, dtype=np.float32)

    good = np.isfinite(phase_deg)
    if not np.any(good):
        return out

    denom = phase_max_deg - phase_min_deg
    if denom <= 0:
        raise ValueError("phase_max_deg must be greater than phase_min_deg")

    out[good] = (phase_deg[good] - phase_min_deg) / denom
    out = np.clip(out, 0.0, 1.0)

    if phase_gamma != 1.0:
        out[good] = np.power(out[good], phase_gamma)

    out[~good] = 0.0
    return out


def make_spectral_colormap(n: int = 2048) -> LinearSegmentedColormap:
    """
    Create a broad spectral FLIM colormap.

    This colormap spans a wide perceptual range from red to violet and is
    intended for general-purpose phase/lifetime rendering when a full
    phase spectrum is desired.

    Parameters
    ----------
    n : int, default=2048
        Number of interpolation levels in the colormap.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        Spectral FLIM colormap.
    """
    colors = [
        (0.00, "#C80000"),
        (0.03, "#E60000"),
        (0.07, "#FF2A00"),
        (0.11, "#FF5500"),
        (0.16, "#FF7A00"),
        (0.21, "#FF9E00"),
        (0.27, "#FFB800"),
        (0.33, "#FFC000"),
        (0.40, "#FFE000"),
        (0.47, "#FFF200"),
        (0.54, "#D8EC00"),
        (0.61, "#8FD400"),
        (0.69, "#32BE32"),
        (0.76, "#00B050"),
        (0.83, "#00A0C8"),
        (0.89, "#0080FF"),
        (0.94, "#304FFE"),
        (0.97, "#3A00FF"),
        (1.00, "#6A00FF"),
    ]
    return LinearSegmentedColormap.from_list("spectral_flim", colors, N=n)


def make_reds_to_greens_colormap(n: int = 2048) -> LinearSegmentedColormap:
    """
    Create a red-to-green FLIM colormap.

    This scale is intended for green-detector FLIM visualization, where a
    restricted warm-to-green progression may be more appropriate than a
    full spectral rainbow.

    Parameters
    ----------
    n : int, default=2048
        Number of interpolation levels in the colormap.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        Red-to-green FLIM colormap.
    """
    colors = [
        (0.00, "#FCF8F8"),  # white with a hint of red
        (0.03, "#FBECEC"),  # very pale pink
        (0.06, "#F9DADA"),  # light pink
        (0.09, "#F6C4C4"),  # soft pink
        (0.12, "#F1A8A8"),  # warm pale red
        (0.15, "#E60000"),  # red
        (0.20, "#FF2A00"),  # red-orange
        (0.25, "#FF5500"),  # orange-red
        (0.30, "#FF7A00"),  # strong orange
        (0.35, "#FF9E00"),  # orange
        (0.40, "#FFB800"),  # orange-yellow
        (0.50, "#FFC000"),  # yellow-orange
        (0.55, "#FFE000"),  # warm yellow
        (0.60, "#FFF200"),  # yellow
        (0.65, "#D8EC00"),  # yellow-lime
        (0.70, "#8FD400"),  # green-yellow
        (0.80, "#00B050"),  # green
        (1.00, "#00A54A"),  # keep green at the top
    ]
    return LinearSegmentedColormap.from_list("reds_to_greens_flim", colors, N=n)


def make_blues_to_greens_colormap(n: int = 2048) -> LinearSegmentedColormap:
    """
    Create a blue-to-green FLIM colormap.

    This scale is intended for blue-detector FLIM visualization and is
    complementary to ``reds_to_greens``. It maps lower phase values to
    pale blue / blue / cyan tones and higher phase values to green tones.

    Parameters
    ----------
    n : int, default=2048
        Number of interpolation levels in the colormap.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        Blue-to-green FLIM colormap.
    """
    colors = [
        (0.00, "#F7FBFF"),  # almost white with blue hint
        (0.03, "#EAF4FF"),  # very pale blue
        (0.06, "#D6ECFF"),  # pale blue
        (0.10, "#BBDFFF"),  # light blue
        (0.15, "#8FCBFF"),  # soft blue
        (0.20, "#5AB3FF"),  # blue
        (0.25, "#2D9CFF"),  # stronger blue
        (0.30, "#0080FF"),  # vivid blue
        (0.36, "#0096D6"),  # blue-cyan
        (0.42, "#00A6C8"),  # cyan-blue
        (0.48, "#00B8B0"),  # cyan-teal
        (0.55, "#00C896"),  # teal-green
        (0.62, "#00D27A"),  # greenish teal
        (0.70, "#3BD35F"),  # light green
        (0.80, "#00B050"),  # green
        (1.00, "#00A54A"),  # keep green at the top
    ]
    return LinearSegmentedColormap.from_list("blues_to_greens_flim", colors, N=n)


def get_phase_colormap(
    name: ScaleName = "spectral",
    n: int = 2048,
) -> LinearSegmentedColormap:
    """
    Return a named FLIM phase colormap.

    Parameters
    ----------
    name : {"spectral", "reds_to_greens", "blues_to_greens"}, default="spectral"
        Name of the requested colormap.
    n : int, default=2048
        Number of interpolation levels in the colormap.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        Requested colormap object.

    Raises
    ------
    ValueError
        If an unknown colormap name is requested.
    """
    if name == "spectral":
        return make_spectral_colormap(n=n)
    if name == "reds_to_greens":
        return make_reds_to_greens_colormap(n=n)
    if name == "blues_to_greens":
        return make_blues_to_greens_colormap(n=n)

    raise ValueError(f"Unknown colormap name: {name}")


def phase_to_rgb(
    phase_deg: np.ndarray,
    scale: ScaleName = "spectral",
    phase_min_deg: float = 0.0,
    phase_max_deg: float = 55.0,
    phase_gamma: float = 0.6,
    n: int = 2048,
) -> np.ndarray:
    """
    Convert phase values in degrees directly into RGB colors.

    This function is useful when the phase has already been computed and only
    colorization is needed.

    Parameters
    ----------
    phase_deg : numpy.ndarray
        Phase array in degrees.
    scale : {"spectral", "reds_to_greens", "blues_to_greens"}, default="spectral"
        Colormap family used for rendering.
    phase_min_deg : float, default=0.0
        Lower phase bound for color mapping.
    phase_max_deg : float, default=55.0
        Upper phase bound for color mapping.
    phase_gamma : float, default=0.6
        Gamma correction applied to the normalized phase.
    n : int, default=2048
        Number of interpolation levels in the colormap.

    Returns
    -------
    numpy.ndarray
        RGB array with shape ``(..., 3)`` and values in [0, 1].
    """
    phase_norm = map_phase_deg_to_norm(
        phase_deg=phase_deg,
        phase_min_deg=phase_min_deg,
        phase_max_deg=phase_max_deg,
        phase_gamma=phase_gamma,
    )

    cmap = get_phase_colormap(name=scale, n=n)
    return cmap(phase_norm)[..., :3]


def phase_intensity_to_rgb(
    g: np.ndarray,
    s: np.ndarray,
    intensity: np.ndarray,
    scale: ScaleName = "spectral",
    phase_min_deg: float = 0.0,
    phase_max_deg: float = 55.0,
    phase_gamma: float = 0.6,
    intensity_gamma: float = 0.99,
    intensity_pmin: float = 1.0,
    intensity_pmax: float = 99.0,
    n: int = 2048,
) -> np.ndarray:
    """
    Convert phasor coordinates and intensity into an RGB pseudocolor image.

    This function computes the phasor phase as:

        phase = atan2(s, g)

    converts that phase to degrees, maps it to a chosen FLIM colormap, and
    modulates the resulting hue by normalized intensity to control brightness.

    Parameters
    ----------
    g : numpy.ndarray
        Real component of the phasor coordinates.
    s : numpy.ndarray
        Imaginary component of the phasor coordinates.
    intensity : numpy.ndarray
        Intensity-like image used for brightness modulation. This is typically
        the mean signal, photon count, or another positive-valued intensity map.
    scale : {"spectral", "reds_to_greens", "blues_to_greens"}, default="spectral"
        Colormap family used for rendering.
    phase_min_deg : float, default=0.0
        Lower phase bound for color mapping.
    phase_max_deg : float, default=55.0
        Upper phase bound for color mapping.
    phase_gamma : float, default=0.6
        Gamma correction applied to normalized phase values.
    intensity_gamma : float, default=0.99
        Gamma correction applied to normalized intensity values.
    intensity_pmin : float, default=1.0
        Lower percentile used for intensity normalization.
    intensity_pmax : float, default=99.0
        Upper percentile used for intensity normalization.
    n : int, default=2048
        Number of interpolation levels in the colormap.

    Returns
    -------
    numpy.ndarray
        RGB image with shape ``(..., 3)`` and values in [0, 1].

    Notes
    -----
    - Invalid pixels are rendered as black.
    - Only pixels satisfying finite ``g``, finite ``s``, finite ``intensity``,
      and ``intensity > 0`` are considered valid.
    - This function is intended for visualization, not for quantitative lifetime
      fitting or direct physical parameter estimation.
    """
    g = np.asarray(g, dtype=np.float32)
    s = np.asarray(s, dtype=np.float32)
    intensity = np.asarray(intensity, dtype=np.float32)

    valid = np.isfinite(g) & np.isfinite(s) & np.isfinite(intensity) & (intensity > 0)

    phase_rad = np.full_like(g, np.nan, dtype=np.float32)
    phase_rad[valid] = np.arctan2(s[valid], g[valid])
    phase_deg = phase_rad_to_deg(phase_rad)

    rgb = phase_to_rgb(
        phase_deg=phase_deg,
        scale=scale,
        phase_min_deg=phase_min_deg,
        phase_max_deg=phase_max_deg,
        phase_gamma=phase_gamma,
        n=n,
    )

    intensity_norm = normalize_percentile(
        intensity,
        pmin=intensity_pmin,
        pmax=intensity_pmax,
    )

    if intensity_gamma != 1.0:
        intensity_norm = intensity_norm**intensity_gamma

    rgb *= intensity_norm[..., None]
    rgb[~valid] = 0.0

    return np.clip(rgb, 0.0, 1.0)


def make_phasor_background(
    scale: ScaleName = "spectral",
    phase_min_deg: float = 0.0,
    phase_max_deg: float = 65.0,
    phase_gamma: float = 1,
    reverse_colors: bool = False,
    background_blend: float = 0.45,
    nx: int = 700,
    ny: int = 450,
    n: int = 2048,
) -> np.ndarray:
    """
    Generate a colored phasor background image.

    This function builds a colored background for the phasor plot by computing
    the phase for each coordinate in the phasor plane (g, s) and mapping that
    phase to a chosen FLIM colormap.

    Only the region inside the universal semicircle is colored. All pixels
    outside the semicircle are left white.

    Parameters
    ----------
    scale : {"spectral", "reds_to_greens", "blues_to_greens"}, default="spectral"
        Colormap family used to render phase values.
    phase_min_deg : float, default=0.0
        Minimum phase (degrees) mapped to the start of the colormap.
    phase_max_deg : float, default=65.0
        Maximum phase (degrees) mapped to the end of the colormap.
    phase_gamma : float, default=0.6
        Gamma correction applied to the normalized phase values.
    nx : int, default=700
        Horizontal resolution of the background image.
    ny : int, default=450
        Vertical resolution of the background image.
    n : int, default=2048
        Number of interpolation levels in the colormap.

    Returns
    -------
    numpy.ndarray
        RGB image with shape (ny, nx, 3) representing the phasor background.

    Notes
    -----
    The universal semicircle corresponds to:

        (g - 0.5)^2 + s^2 <= 0.25

    which represents single-exponential lifetimes in phasor space.
    """

    g = np.linspace(0.0, 1.0, nx)
    s = np.linspace(0.0, 0.65, ny)
    gg, ss = np.meshgrid(g, s)

    phase_rad = np.arctan2(ss, gg)
    phase_deg = phase_rad_to_deg(phase_rad)

    inside_phase_range = (
        (phase_deg >= phase_min_deg)
        & (phase_deg <= phase_max_deg)

    )

    inside_semicircle = (

        ((gg - 0.5) ** 2 + ss**2 <= 0.25)
        & (ss >= 0)
        & inside_phase_range

    )

    phase_norm = map_phase_deg_to_norm(
        phase_deg,
        phase_min_deg,
        phase_max_deg,
        phase_gamma,

    )

    if reverse_colors:
        phase_norm = 1.0 - phase_norm
    cmap = get_phase_colormap(scale, n=n)
    rgb = np.ones((ny, nx, 3), dtype=np.float32)
    rgb_inside = cmap(phase_norm)[..., :3].astype(np.float32)
    background_blend = float(np.clip(background_blend, 0.0, 1.0))
    rgb_inside = rgb_inside * (1.0 - background_blend) + background_blend
    rgb[inside_semicircle] = rgb_inside[inside_semicircle]

    return rgb



def make_phase_legend_strip(
    scale: ScaleName = "spectral",
    phase_min_deg: float = 0.0,
    phase_max_deg: float = 65.0,
    phase_gamma: float = 0.6,
    width: int = 1024,
    height: int = 80,
    horizontal: bool = True,
    n: int = 2048,
) -> np.ndarray:
    """
    Generate a phase color legend strip.

    This function creates a color bar image representing the mapping between
    phase values in degrees and the corresponding RGB colors used in FLIM
    pseudocolor visualization.

    Parameters
    ----------
    scale : {"spectral", "reds_to_greens", "blues_to_greens"}, default="spectral"
        Colormap family used to render phase values.
    phase_min_deg : float, default=0.0
        Minimum phase represented in the legend.
    phase_max_deg : float, default=65.0
        Maximum phase represented in the legend.
    phase_gamma : float, default=0.6
        Gamma correction applied to the phase mapping.
    width : int, default=1024
        Width of the legend image in pixels.
    height : int, default=80
        Height of the legend image in pixels.
    horizontal : bool, default=True
        Orientation of the legend strip.
        If False, the strip is vertical.
    n : int, default=2048
        Number of interpolation levels in the colormap.

    Returns
    -------
    numpy.ndarray
        RGB image containing the color legend.

    Notes
    -----
    The legend represents the same phase-to-color mapping used by
    ``phase_to_rgb`` and ``phase_intensity_to_rgb``.
    """
    if horizontal:
        phase_vals = np.linspace(phase_min_deg, phase_max_deg, width)
        phase_vals = np.tile(phase_vals, (height, 1))
    else:
        phase_vals = np.linspace(phase_min_deg, phase_max_deg, height)
        phase_vals = np.tile(phase_vals[:, None], (1, width))

    rgb = phase_to_rgb(
        phase_deg=phase_vals,
        scale=scale,
        phase_min_deg=phase_min_deg,
        phase_max_deg=phase_max_deg,
        phase_gamma=phase_gamma,
        n=n,
    )

    return rgb
