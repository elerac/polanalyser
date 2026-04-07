from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure


def imshow_grid(
    images: Sequence[Sequence[npt.NDArray[np.uint8] | None]],
    cmaps: Sequence[Sequence[str | Colormap | None]] | None = None,
    norms: Sequence[Sequence[Normalize | None]] | None = None,
    labels: Sequence[Sequence[str | None]] | None = None,
    *,
    figw: float = 6.4,
    gap: float = 0.05,
    cbar_h: float = 0.08,
    cbar_gap: float = 0.03,
    title_fontsize: float = 10.0,
    cbar_tick_fontsize: float = 10.0,
) -> Figure:
    """Plot a grid of images with per-panel horizontal colorbars.

    Renders a compact, uniform mosaic of RGB images arranged on a regular
    grid.  Each panel shows its image with the supplied colormap /
    normalization and gets a dedicated horizontal colorbar directly
    beneath it.  All spacing (inter-panel gaps and outer margins) is
    equal, and the figure size is computed automatically so that nothing
    overlaps.

    Parameters
    ----------
    images : Sequence of Sequence of (ndarray or None), shape (H_GRID, W_GRID) of (H, W, 3) uint8
        2-D sequence of color images in BGR order.  ``images[r][c]`` is
        the image for row *r*, column *c*.  All non-``None`` images must
        share the same (H, W) shape. ``None`` entries produce a blank (empty) panel.
    cmaps : Sequence of Sequence of (str or Colormap or None) or None, optional
        Per-panel colormaps, same grid shape as *images*.  If an entry
        is ``None``, no colorbar is drawn for that panel.  If *all*
        entries in a row are ``None``, the colorbar space for that row
        is reclaimed so the layout stays tight.  If the entire argument
        is ``None``, no colorbars are drawn anywhere.
    norms : Sequence of Sequence of (Normalize or None) or None, optional
        Per-panel normalizations (carry ``vmin`` / ``vmax``), same grid
        shape as *images*.  Should be ``None`` where the corresponding
        *cmaps* entry is ``None``.  If the entire argument is ``None``,
        no normalizations are applied.
    labels : Sequence of Sequence of str or None, optional
        Per-panel title strings.  If ``None``, no titles are drawn and
        no vertical space is reserved for them.
    figw : float, default 6.4
        Overall figure width in inches.
    gap : float, default 0.05
        Uniform gap in inches used for inter-panel spacing *and* outer
        margins, ensuring visually consistent whitespace everywhere.
    cbar_h : float, default 0.08
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
        The fully laid-out figure.  Call ``fig.savefig(...)`` to export.
    """
    h_grid = len(images)
    w_grid = len(images[0])
    if any(len(row) != w_grid for row in images):
        raise ValueError("images must be a rectangular 2D sequence")
    if all(images[r][c] is None for r in range(h_grid) for c in range(w_grid)):
        raise ValueError("At least one image must be non-None")

    # Default all-None grids for cmaps / norms / labels when passed as None.
    if cmaps is None:
        cmaps = [[None] * w_grid for _ in range(h_grid)]
    if norms is None:
        norms = [[None] * w_grid for _ in range(h_grid)]
    if labels is None:
        labels = [[None] * w_grid for _ in range(h_grid)]

    if len(cmaps) != h_grid or any(len(row) != w_grid for row in cmaps):
        raise ValueError("cmaps must match images grid shape")
    if len(norms) != h_grid or any(len(row) != w_grid for row in norms):
        raise ValueError("norms must match images grid shape")
    if len(labels) != h_grid or any(len(row) != w_grid for row in labels):
        raise ValueError("labels must match images grid shape")

    # Find the first non-None image to determine pixel dimensions.
    _ref_img = next(img for row in images for img in row if img is not None)
    img_h_px, img_w_px = _ref_img.shape[:2]

    has_any_label = any(lbl is not None for row in labels for lbl in row)

    # Reserve space for title and colorbar ticks (tuned for the font sizes).
    title_reserve = (title_fontsize / 72) if has_any_label else 0.0
    tick_reserve = cbar_tick_fontsize / 72 + 0.1

    # Use Matplotlib's default figure width and derive panel width from it.
    panel_w_eff = (figw - (w_grid + 1) * gap) / w_grid
    if panel_w_eff <= 0:
        raise ValueError(f"gap is too large to fit the grid within figw={figw}")

    # Image height preserving the pixel aspect ratio
    img_h = panel_w_eff * img_h_px / img_w_px

    # Per-row: check whether any panel in the row has a colorbar
    row_has_cbar = [any(cmaps[r][c] is not None and images[r][c] is not None for c in range(w_grid)) for r in range(h_grid)]

    # Per-row group height (skip colorbar space when the entire row lacks colorbars)
    group_h_full = title_reserve + img_h + cbar_gap + cbar_h + tick_reserve
    group_h_nocbar = title_reserve + img_h
    group_heights = [group_h_full if row_has_cbar[r] else group_h_nocbar for r in range(h_grid)]

    # Figure dimensions (uniform outer margins = gap)
    figh = (h_grid + 1) * gap + sum(group_heights)

    fig = plt.figure(figsize=(figw, figh))

    # Cumulative y position from figure top
    y_cursor = figh  # start at top

    for r in range(h_grid):
        y_cursor -= gap  # top margin / inter-row gap
        gh = group_heights[r]

        for c in range(w_grid):
            # Horizontal position (inches from left)
            x = gap + c * (panel_w_eff + gap)

            # Vertical position (inches from bottom; row 0 = top)
            y_img_bottom = y_cursor - title_reserve - img_h
            y_cbar_bottom = y_img_bottom - cbar_gap - cbar_h

            # Add axes
            ax = fig.add_axes((x / figw, y_img_bottom / figh, panel_w_eff / figw, img_h / figh))
            ax.set_xticks([])
            ax.set_yticks([])

            # Draw image if present
            image_rc = images[r][c]
            if image_rc is not None:
                # Display image
                ax.imshow(image_rc[..., ::-1])  # RGB -> BGR for Matplotlib

                # Panel title
                label_rc = labels[r][c]
                if label_rc is not None:
                    ax.set_title(label_rc, fontsize=title_fontsize, pad=2)

                # Colorbar axes (skip if cmap is None)
                cmap_rc = cmaps[r][c]
                norm_rc = norms[r][c]
                if cmap_rc is not None:
                    ax_cbar = fig.add_axes((x / figw, y_cbar_bottom / figh, panel_w_eff / figw, cbar_h / figh))
                    sm = ScalarMappable(cmap=cmap_rc, norm=norm_rc)
                    sm.set_array([])
                    cb: Colorbar = fig.colorbar(sm, cax=ax_cbar, orientation="horizontal")
                    cb.ax.tick_params(labelsize=cbar_tick_fontsize)
                    vmin, vmax = float(sm.norm.vmin), float(sm.norm.vmax)  # type: ignore
                    cb.set_ticks([vmin, (vmin + vmax) / 2, vmax])
                    cb.set_ticklabels([f"{vmin:.2f}", f"{(vmin + vmax) / 2:.2f}", f"{vmax:.2f}"])
                    xticklabels = cb.ax.get_xticklabels()
                    if xticklabels:
                        xticklabels[0].set_horizontalalignment("left")
                        xticklabels[-1].set_horizontalalignment("right")
            else:
                # No image: hide axes
                for spine in ax.spines.values():
                    spine.set_visible(False)

        y_cursor -= gh  # advance past this row's content

    return fig
