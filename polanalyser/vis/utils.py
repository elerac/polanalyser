import numpy as np

from ..stokes import *
from .colorize import ColorizerSpec
from .polar_colorize import (
    _colorize_aolp_spec,
    _colorize_cop_spec,
    _colorize_dop_spec,
    _colorize_stokes_spec,
    _colorize_top_spec,
)

# Default presets
OUTPUTS_DEFAULT = [
    ["s0", "s1", "s2", "s3"],
    ["aolp", "dop", "cop", "top"],
]
OUTPUTS_DEFAULT_LINEAR = ["s0", "s1", "s2", "dolp", "aolp"]


def _guess_stokes_properties(img_stokes: np.ndarray) -> tuple[bool, bool]:
    """Infer (is_linear, is_colored) from Stokes image shape.

    - (H, W, 3, 3): Colored Linear-only Stokes image
    - (H, W, 3, 4): Colored Stokes image
    - (H, W, 3): Linear-only Stokes image
    - (H, W, 4): Stokes image
    """
    if img_stokes.ndim < 2:
        raise ValueError(f"Expected img_stokes to have at least 2 dimensions, got {img_stokes.shape}")
    if img_stokes.ndim >= 4 and img_stokes.shape[2] == 3 and img_stokes.shape[3] == 3:
        return True, True
    if img_stokes.ndim >= 4 and img_stokes.shape[2] == 3 and img_stokes.shape[3] == 4:
        return False, True
    if img_stokes.ndim >= 3 and img_stokes.shape[2] == 3:
        return True, False
    if img_stokes.ndim >= 3 and img_stokes.shape[2] == 4:
        return False, False
    raise ValueError("Expected img_stokes to have shape (H, W, 3, 3) or (H, W, 3, 4) or (H, W, 3) or (H, W, 4), " f"got {img_stokes.shape}")


def _resolve_stokes_outputs(img_stokes: np.ndarray) -> tuple[list[list[str]], list[str]]:
    is_linear, _ = _guess_stokes_properties(img_stokes)
    if is_linear:
        output_grid = [list(OUTPUTS_DEFAULT_LINEAR)]
    else:
        output_grid = [list(row) for row in OUTPUTS_DEFAULT]

    output_grid = [[str(item) for item in row] for row in output_grid]
    output_keys = [str(key) for row in output_grid for key in row]
    return output_grid, output_keys


def _render_stokes_spec(
    img_stokes: np.ndarray,
    gamma: float = 1.0,
    halfrange: float | None = None,
) -> dict[str, tuple[np.ndarray, ColorizerSpec]]:
    """Convert Stokes vector to rendered outputs and colorizer specs."""
    img_stokes = np.asarray(img_stokes)

    # Determine outputs
    _, is_colored = _guess_stokes_properties(img_stokes)
    _, output_keys = _resolve_stokes_outputs(img_stokes)

    # Determine halfrange
    if halfrange is None:
        halfrange = np.nanmax(np.abs(img_stokes))

    # For colored Stokes, convert to mono-Stokes for parameter calculations
    img_bgr_stokes = None
    if is_colored:
        img_bgr_stokes = img_stokes
        img_stokes = np.average(img_bgr_stokes, axis=2)

    # Render each requested output
    results: dict[str, tuple[np.ndarray, ColorizerSpec]] = {}
    for key in output_keys:
        if key == "aolp":
            aolp = stokes_to_aolp(img_stokes)
            aolp_colored, colorizer_aolp = _colorize_aolp_spec(aolp)
            results[key] = (aolp_colored, colorizer_aolp)
        elif key == "dop":
            dop = stokes_to_dop(img_stokes)
            dop_colored, colorizer_dop = _colorize_dop_spec(dop)
            results[key] = (dop_colored, colorizer_dop)
        elif key == "dolp":
            dolp = stokes_to_dolp(img_stokes)
            dolp_colored, colorizer_dolp = _colorize_dop_spec(dolp)
            results[key] = (dolp_colored, colorizer_dolp)
        elif key == "docp":
            docp = stokes_to_docp(img_stokes)
            docp_colored, colorizer_docp = _colorize_dop_spec(docp)
            results[key] = (docp_colored, colorizer_docp)
        elif key == "cop":
            eang = stokes_to_eang(img_stokes)
            docp = stokes_to_docp(img_stokes)
            cop_colored, colorizer_cop = _colorize_cop_spec(eang, docp)
            results[key] = (cop_colored, colorizer_cop)
        elif key == "top":
            eang = stokes_to_eang(img_stokes)
            dop = stokes_to_dop(img_stokes)
            top_colored, colorizer_top = _colorize_top_spec(eang, dop)
            results[key] = (top_colored, colorizer_top)
        elif key in ["s0", "s1", "s2", "s3"]:
            if key == "s0" and img_bgr_stokes is not None:
                # Special case: extract s0 from colored Stokes image
                img_bgr_s0 = img_bgr_stokes[..., 0]
                si_colored = np.clip((img_bgr_s0 / halfrange) ** gamma * 255.0, 0, 255).astype(np.uint8)
                colorizer_si = ColorizerSpec()
            else:
                i = {"s0": 0, "s1": 1, "s2": 2, "s3": 3}[key]
                si = img_stokes[..., i]
                si_colored, colorizer_si = _colorize_stokes_spec(si, gamma=gamma, halfrange=halfrange)
            results[key] = (si_colored, colorizer_si)
        else:
            raise ValueError(f"Unsupported key: {key}")

    return results


def render_stokes(
    img_stokes: np.ndarray,
    gamma: float = 1.0,
    halfrange: float | None = None,
) -> dict[str, np.ndarray]:
    """Render Stokes components and derived parameters as color images.

    Parameters
    ----------
    img_stokes : ndarray
        Stokes image with shape (H, W, 4), (H, W, 3), (H, W, 3, 4), or
        (H, W, 3, 3). The four-dimensional variants represent color Stokes.
    gamma : float, default 1.0
        Gamma for Stokes intensity normalization.
    halfrange : float, optional
        Half of the symmetric data range for s0/s1/s2/s3 visualization.
        If None, computed from the input.

    Returns
    -------
    images : dict[str, ndarray]
        Mapping from output key to colorized image (uint8, BGR order).

    Examples
    --------
    Render the default preset from a Stokes image (H, W, 4):

    >>> results = pa.render_stokes(img_stokes)
    >>> results.keys()
    dict_keys(['s0', 's1', 's2', 's3', 'aolp', 'dop', 'cop', 'top'])
    >>> results["aolp"].shape
    (H, W, 3), uint8
    """
    results = _render_stokes_spec(img_stokes, gamma, halfrange)
    return {key: img for key, (img, _) in results.items()}
