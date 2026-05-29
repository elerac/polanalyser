from typing import Any, Literal

import matplotlib.colorizer
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text

from .imshow import imshow_grid
from .layout import tile_mueller
from .norm import SymPowerNorm
from .utils import _render_stokes_spec, _resolve_stokes_outputs


def imshow_stokes(
    img_stokes: np.ndarray,
    gamma: float = 1.0,
    halfrange: float | None = None,
    *,
    gap: float = 0.05,
    cbar_h: float = 0.08,
    cbar_gap: float = 0.03,
    title_fontsize: float = 10.0,
    cbar_tick_fontsize: float = 10.0,
) -> Figure:
    """Display Stokes components and derived parameters in a grid.

    Parameters
    ----------
    img_stokes : ndarray
        Stokes image with shape (H, W, 4), (H, W, 3), (H, W, 3, 4), or
        (H, W, 3, 3). The four-dimensional variants represent color Stokes.
    gamma : float, default 1.0
        Gamma for stokes intensity normalization.
    halfrange : float, optional
        Half of the symmetric data range for S1/S2/S3 visualization.
        If None, computed from the input.
    gap : float, default 0.1
        Uniform gap in inches used for inter-panel spacing *and* outer
        margins, ensuring visually consistent whitespace everywhere.
    cbar_h : float, default 0.10
        Height of each colorbar in inches.
    cbar_gap : float, default 0.03
        Vertical gap between the bottom of the image and the top of the
        colorbar, in inches.
    title_fontsize : float, default 10
        Font size (in points) for the per-panel titles.
    cbar_tick_fontsize : float, default 10
        Font size (in points) for the colorbar tick labels.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The rendered figure.
    """
    img_stokes = np.asarray(img_stokes)

    output_grid, _ = _resolve_stokes_outputs(img_stokes)
    results = _render_stokes_spec(img_stokes, gamma, halfrange)

    images: list[list[np.ndarray | None]] = []
    cmaps: list[list[str | matplotlib.colors.Colormap | None]] = []
    norms: list[list[matplotlib.colors.Normalize | None]] = []
    labels: list[list[str | None]] = []

    for row in output_grid:
        images_row: list[np.ndarray | None] = []
        cmaps_row: list[str | matplotlib.colors.Colormap | None] = []
        norms_row: list[matplotlib.colors.Normalize | None] = []
        labels_row: list[str | None] = []
        for key in row:
            img_bgr, colorizer = results[key]
            images_row.append(img_bgr)
            cmaps_row.append(colorizer.cmap)
            norms_row.append(colorizer.norm)
            labels_row.append(key)
        images.append(images_row)
        cmaps.append(cmaps_row)
        norms.append(norms_row)
        labels.append(labels_row)

    return imshow_grid(
        images,
        cmaps,
        norms,
        labels,
        gap=gap,
        cbar_h=cbar_h,
        cbar_gap=cbar_gap,
        title_fontsize=title_fontsize,
        cbar_tick_fontsize=cbar_tick_fontsize,
    )


def imshow_mueller(
    mueller: npt.ArrayLike,
    *,
    ax: Axes | None = None,
    gamma: float = 1.0,
    halfrange: float | None = None,
    text: Literal["mean", "average", "median", "min", "max", "std", "var", "nanmean", "nanmedian", "nanmin", "nanmax", "nanstd", "nanvar"] | None = None,
    line_kw: dict[str, Any] | None = None,
    text_kw: dict[str, Any] | None = None,
    **kwargs,
):
    """Display a Mueller matrix as a tiled grid.

    Parameters
    ----------
    mueller : arra-like, (4, 4) or (H, W, 4, 4) or (H, W, 3, 4, 4)
        A single Mueller matrix (4, 4) or a Mueller matrix image (H, W, 4, 4) or a RGB Mueller matrix image (H, W, 3, 4, 4).
    ax : Axes, optional
        Axes to draw on. If None, uses the current axes (plt.gca()).
    gamma : float, optional
        Gamma for SymPowerNorm. Default: 1.0.
    halfrange : float, optional
        Half of the symmetric data range for the colormap. Actual limits are vmin=-halfrange and vmax=halfrange.
    text : {"mean", "average", "median", "min", "max", "std", "var", "nanmean", "nanmedian", "nanmin", "nanmax", "nanstd", "nanvar"} or None, optional
        If specified, overlay the per-tile statistic at the center of each tile.
        None disables text. Default: None.
    line_kw : dict, optional
        Additional keyword arguments forwarded to Axes.axhline/axvline for grid lines.
    text_kw : dict, optional
        Additional keyword arguments forwarded to Axes.text for overlay text.
    **kwargs
        Additional keyword arguments forwarded to Axes.imshow or Axes.pcolormesh.

    Returns
    -------
    image_artist : ColorizingArtist
        The image artist created by Axes.imshow or Axes.pcolormesh.
    line_artists : list of Line2D
        List of line artists for the grid lines.
    text_artists : list of Text
        List of text artists for the overlay text.
    """
    ax = ax or plt.gca()

    # If halfrange is specified, ensure symmetric range.
    vmin = vmax = None
    if halfrange is not None:
        vmin = -halfrange
        vmax = halfrange

    # Default cmap and norm
    cmap = "mueller"
    norm = SymPowerNorm(gamma=gamma)
    colorizer = matplotlib.colorizer.Colorizer(cmap=cmap, norm=norm)
    colorizer.set_clim(vmin=vmin, vmax=vmax)

    # Check shape of Mueller matrix
    mueller = np.asarray(mueller)
    if mueller.ndim == 2:
        # Single matrix case: (4, 4)
        is_matrix = True
    elif mueller.ndim == 4 or (mueller.ndim == 5 and mueller.shape[2] == 3):
        # Mono-Mueller matrix image case: (H, W, 4, 4)
        # Color-Mueller matrix image case: (H, W, 3, 4, 4)
        is_matrix = False
    else:
        raise ValueError(f"mueller must have shape (H, W, 4, 4) or (H, W, 3, 4, 4) or (4, 4), but got {mueller.shape}")

    # Display Mueller matrix grid
    M1, M2 = mueller.shape[-2:]
    if is_matrix:
        # Single matrix case: use pcolormesh
        # I found that pcolormesh is robust for rendering a single cell.
        x_centers = np.arange(M2)
        y_centers = np.arange(M1)
        image_artist = ax.pcolormesh(x_centers, y_centers, mueller, colorizer=colorizer, edgecolors="none", **kwargs)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks(x_centers)
        ax.set_yticks(y_centers)
        ax._sci(image_artist)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
    else:
        # Image case: use imshow
        img = tile_mueller(mueller)
        image_artist = ax.imshow(img, colorizer=colorizer, **kwargs)
        ax._sci(image_artist)
        x0, x1, y0, y1 = image_artist.get_extent()

    cell_h = (y1 - y0) / M1
    cell_w = (x1 - x0) / M2

    # Draw grid lines by axhline/axvline
    line_kw_preset = {
        "linewidth": plt.rcParams.get("axes.linewidth", 0.8),
        "color": plt.rcParams.get("axes.edgecolor", "black"),
    }
    if line_kw is not None:
        line_kw_preset.update(line_kw)
    linewidth = line_kw_preset.pop("linewidth")  # Pop linewidth to scale it for border lines
    line_artists: list[Line2D] = []
    bbox = Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, transform=ax.transData)
    for i in range(0, M2 + 1):
        scale = 2.0 if i == 0 or i == M2 else 1.0  # Thicker border lines for outer edges
        line = ax.axvline(x0 + i * cell_w, linewidth=linewidth * scale, **line_kw_preset)
        line.set_clip_path(bbox)
        line_artists.append(line)
    for j in range(0, M1 + 1):
        scale = 2.0 if j == 0 or j == M1 else 1.0  # Thicker border lines for outer edges
        line = ax.axhline(y0 + j * cell_h, linewidth=linewidth * scale, **line_kw_preset)
        line.set_clip_path(bbox)
        line_artists.append(line)

    # Hide spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Overlay text if specified
    text_artists: list[Text] = []
    if text is not None:
        text_kw_preset = {
            "ha": "center",
            "va": "center",
            "bbox": {
                "boxstyle": "round,pad=0.2",
                "facecolor": plt.rcParams.get("figure.facecolor", "white"),
                "edgecolor": "none",
                "alpha": 0.7,
            },
        }
        if text_kw is not None:
            text_kw_preset.update(text_kw)

        reduce_func = getattr(np, text)
        if is_matrix:
            mueller = mueller[None, None, :, :]  # (1, 1, 4, 4)
        mueller_reduced = reduce_func(mueller, axis=(0, 1))  # (4, 4)

        x_centers = np.linspace(x0 + 0.5 * cell_w, x1 - 0.5 * cell_w, M2)
        y_centers = np.linspace(y0 + 0.5 * cell_h, y1 - 0.5 * cell_h, M1)[::-1]
        for j, yj in enumerate(y_centers):
            for i, xi in enumerate(x_centers):
                s = f"{mueller_reduced[j, i]:.3f}"
                if s == "-0.000":
                    s = "0.000"
                t = ax.text(xi, yj, s, **text_kw_preset)
                text_artists.append(t)

    return image_artist, line_artists, text_artists
