from dataclasses import dataclass
from typing import Literal, Sequence

import matplotlib
import matplotlib.colorizer
import matplotlib.colors
import numpy as np
import numpy.typing as npt


@dataclass
class ColorizerSpec:
    """Specification for a colorizer. (Avoid using matplotlib.colorizer.Colorizer directly)."""

    cmap: str | matplotlib.colors.Colormap | None = None
    norm: matplotlib.colors.Normalize | None = None


def _colorize_spec(
    x: np.ndarray,
    cmap: str | matplotlib.colors.Colormap | None = None,
    norm: matplotlib.colors.Normalize | None = None,
    *,
    colorizer: matplotlib.colorizer.Colorizer | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    sat_gain: float | npt.NDArray[np.floating] | None = None,
    val_gain: float | npt.NDArray[np.floating] | None = None,
    nan_color: int | Sequence[int] | npt.NDArray[np.uint8] = 255,
    color_order: Literal["bgr", "rgb"] = "bgr",
) -> tuple[npt.NDArray[np.uint8], ColorizerSpec]:
    """Apply a Matplotlib colormap to an array and return both colorized output and a colorizer spec."""
    x = np.asarray(x)

    if colorizer is None:
        # Create colorizer if not provided
        colorizer = matplotlib.colorizer.Colorizer(cmap=cmap, norm=norm)
        colorizer.set_clim(vmin=vmin, vmax=vmax)
    else:
        # Check conflicting parameters
        conflicted_params = {"cmap": cmap, "norm": norm, "vmin": vmin, "vmax": vmax}
        if any([val is not None for val in conflicted_params.values()]):
            raise ValueError("The `colorizer` keyword cannot be used simultaneously" " with any of the following keywords: " + ", ".join(f"`{key}`" for key in conflicted_params.keys()))

    # Apply colorizer
    if colorizer.norm is not None:
        x = colorizer.norm(x)
    x_rgb = colorizer.cmap(x)[..., :3]  # (..., 3) np.float32

    # HSV scaling
    if (sat_gain is not None) or (val_gain is not None):
        sat_gain = 1.0 if sat_gain is None else sat_gain
        val_gain = 1.0 if val_gain is None else val_gain
        hsv = matplotlib.colors.rgb_to_hsv(x_rgb)
        hsv[..., 1] *= sat_gain
        hsv[..., 2] *= val_gain
        x_rgb = matplotlib.colors.hsv_to_rgb(hsv.clip(0, 1))

    # Convert float32 [0, 1] to uint8 [0, 255]
    x_rgb = np.clip(x_rgb * 255.0, 0, 255).astype(np.uint8)  # (..., 3), np.uint8

    # Change color channel order if needed
    if color_order == "bgr":
        x_color = x_rgb[..., ::-1]
    elif color_order == "rgb":
        x_color = x_rgb
    else:
        raise ValueError(f"Unknown order={color_order!r}. Use 'bgr' or 'rgb'.")

    # Set NaN color
    x_color[np.isnan(x)] = nan_color

    # Create ColorizerSpec
    colorizer_spec = ColorizerSpec(
        cmap=colorizer.cmap,
        norm=colorizer.norm,
    )

    return x_color, colorizer_spec


def colorize(
    x: np.ndarray,
    cmap: str | matplotlib.colors.Colormap | None = None,
    norm: matplotlib.colors.Normalize | None = None,
    *,
    colorizer: matplotlib.colorizer.Colorizer | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    sat_gain: float | npt.NDArray[np.floating] | None = None,
    val_gain: float | npt.NDArray[np.floating] | None = None,
    nan_color: int | Sequence[int] | npt.NDArray[np.uint8] = 255,
    color_order: Literal["bgr", "rgb"] = "bgr",
) -> npt.NDArray[np.uint8]:
    """Apply a Matplotlib colormap to a given array, with optional HSV scaling.

    Parameters
    ----------
    x : ndarray, (...,)
        Input array.
    cmap : str, matplotlib.colors.Colormap, optional
        Colormap name or Colormap instance, by default None (uses default colormap).
    norm : matplotlib.colors.Normalize, optional
        Normalization instance, by default None (linear normalization).
    colorizer : matplotlib.colorizer.Colorizer, optional
        Colorizer instance. If provided, *cmap* and *norm* are ignored.
    vmin : float, optional
        The minimum value to normalize. If None, the minimum value of *x* is used.
    vmax : float, optional
        The maximum value to normalize. If None, the maximum value of *x* is used.
    sat_gain : float or ndarray, optional
        Gain factor for saturation in HSV color space, by default None (no scaling)
    val_gain : float or ndarray, optional
        Gain factor for value in HSV color space, by default None (no scaling)
    nan_color : int, Sequence[int], npt.NDArray[np.uint8], optional
        The value to set for NaN values, by default 255 (white).
    color_order : {"bgr", "rgb"}, default: "bgr"
        Color channel order for output array.

    Returns
    -------
    x_color : ndarray, (..., 3), uint8
        Colored array of input array.

    Examples
    --------
    Apply "viridis" colormap for a 2D array.

    >>> x = np.random.rand(256, 256)
    >>> x.shape, x.dtype
    (256, 256) float64
    >>> x_bgr = pa.colorize(x, "viridis", vmin=0.0, vmax=1.0)
    >>> x_bgr.shape, x_bgr.dtype
    (256, 256, 3) uint8

    Apply "RdBu" colormap for a 3D array.

    >>> x = 2 * np.random.rand(128, 64, 64) - 1.0
    >>> x.shape, x.dtype
    (128, 64, 64) float64
    >>> x_bgr = pa.colorize(x, "RdBu", vmin=-1.0, vmax=1.0)
    >>> x_bgr.shape, x_bgr.dtype
    (128, 64, 64, 3) uint8

    Apply saturation scaling based on another array.

    >>> x = np.random.rand(256, 256) * np.pi
    >>> sat_gain = np.random.rand(256, 256)
    >>> x_bgr = pa.colorize(x, "hsv", vmin=0.0, vmax=np.pi, sat_gain=sat_gain)

    Apply logarithmic normalization.

    >>> x = np.exp(-10 * np.random.rand(256, 256))
    >>> norm = matplotlib.colors.LogNorm(vmin=x.min(), vmax=x.max())
    >>> x_bgr = pa.colorize(x, cmap="PuBu_r", norm=norm)
    """
    x_color, _ = _colorize_spec(
        x,
        cmap=cmap,
        norm=norm,
        colorizer=colorizer,
        vmin=vmin,
        vmax=vmax,
        sat_gain=sat_gain,
        val_gain=val_gain,
        nan_color=nan_color,
        color_order=color_order,
    )
    return x_color
