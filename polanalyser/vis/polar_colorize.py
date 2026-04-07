import matplotlib
import matplotlib.colors
import numpy as np
import numpy.typing as npt

from .colorize import ColorizerSpec, _colorize_spec
from .colorramp import eval_colorramp
from .norm import SymPowerNorm


def register_polarization_colormaps() -> None:
    """Register polarization colormaps to matplotlib.

    This function is called automatically upon module import.
    """
    red = matplotlib.colors.to_rgb("red")
    blue = matplotlib.colors.to_rgb("blue")
    yellow = matplotlib.colors.to_rgb("yellow")
    cyan = matplotlib.colors.to_rgb("cyan")
    black = matplotlib.colors.to_rgb("black")
    t = np.linspace(0, 1, 256, dtype=np.float32)

    cmap_stokes = matplotlib.colormaps.get_cmap("RdBu")
    matplotlib.colormaps.register(cmap_stokes, name="stokes", force=True)

    cmap_mueller = matplotlib.colormaps.get_cmap("RdBu")
    matplotlib.colormaps.register(cmap_mueller, name="mueller", force=True)

    cmap_aolp = matplotlib.colormaps.get_cmap("hsv")
    matplotlib.colormaps.register(cmap_aolp, name="aolp", force=True)

    lut_dop = eval_colorramp(t, [black, red], space="srgb")
    cmap_dop = matplotlib.colors.ListedColormap(lut_dop)
    matplotlib.colormaps.register(cmap_dop, name="dop", force=True)

    lut_top = eval_colorramp(t, [yellow, cyan, yellow], space="oklab")
    cmap_top = matplotlib.colors.ListedColormap(lut_top)
    matplotlib.colormaps.register(cmap_top, name="top", force=True)

    lut_cop = eval_colorramp(t, [yellow, black, blue], space="oklab")
    cmap_cop = matplotlib.colors.ListedColormap(lut_cop)
    matplotlib.colormaps.register(cmap_cop, name="cop", force=True)


register_polarization_colormaps()


def _colorize_aolp_spec(aolp: np.ndarray, dolp_sat: np.ndarray | None = None, dolp_val: np.ndarray | None = None) -> tuple[npt.NDArray[np.uint8], ColorizerSpec]:
    return _colorize_spec(aolp, cmap="aolp", vmin=0.0, vmax=np.pi, sat_gain=dolp_sat, val_gain=dolp_val)


def colorize_aolp(aolp: np.ndarray, dolp_sat: np.ndarray | None = None, dolp_val: np.ndarray | None = None) -> npt.NDArray[np.uint8]:
    """Apply colormap to AoLP (Angle of Linear Polarization).

    Parameters
    ----------
    aolp : np.ndarray, (...,)
        Angle of Linear Polarization (AoLP) [0.0, pi].
    dolp_sat : np.ndarray, (...,), optional
        Degree of Linear Polarization (DoLP) for saturation scaling [0.0, 1.0].
    dolp_val : np.ndarray, (...,), optional
        Degree of Linear Polarization (DoLP) for value scaling [0.0, 1.0].

    Returns
    -------
    aolp_colored : np.ndarray, (..., 3), uint8
        Colored array of AoLP.
    """
    return _colorize_aolp_spec(aolp, dolp_sat=dolp_sat, dolp_val=dolp_val)[0]


def _colorize_dop_spec(dop: np.ndarray) -> tuple[npt.NDArray[np.uint8], ColorizerSpec]:
    return _colorize_spec(dop, cmap="dop", vmin=0.0, vmax=1.0)


def colorize_dop(dop: np.ndarray) -> npt.NDArray[np.uint8]:
    """Apply colormap to Degree of Polarization (DoP) or Degree of Linear/Circular Polarization (DoLP/DoCP).

    Parameters
    ----------
    dop : np.ndarray, (...,)
        Degree of Polarization (DoP) [0.0, 1.0].

    Returns
    -------
    dop_colored : np.ndarray, (..., 3), uint8
        Colored array of DoP.
    """
    return _colorize_dop_spec(dop)[0]


def _colorize_top_spec(eang: np.ndarray, dop: np.ndarray | None = None) -> tuple[npt.NDArray[np.uint8], ColorizerSpec]:
    return _colorize_spec(eang, cmap="top", vmin=-np.pi / 4, vmax=np.pi / 4, val_gain=dop)


def colorize_top(eang: np.ndarray, dop: np.ndarray | None = None) -> npt.NDArray[np.uint8]:
    """Apply colormap to ToP (Type of Polarization).

    Parameters
    ----------
    eang : np.ndarray, (...,)
        Ellipticity angle [-pi/4, pi/4].
    dop : np.ndarray, (...,), optional
        Degree of Polarization (DoP) [0.0, 1.0].

    Returns
    -------
    top_colored : np.ndarray, (..., 3), uint8
        Colored array of ToP.
    """
    return _colorize_top_spec(eang, dop)[0]


def _colorize_cop_spec(eang: np.ndarray, docp: np.ndarray | None = None) -> tuple[npt.NDArray[np.uint8], ColorizerSpec]:
    return _colorize_spec(eang, cmap="cop", vmin=-np.pi / 4, vmax=np.pi / 4, val_gain=docp)


def colorize_cop(eang: np.ndarray, docp: np.ndarray | None = None) -> npt.NDArray[np.uint8]:
    """Apply colormap to CoP (Chirality of Polarization).

    Parameters
    ----------
    eang : np.ndarray, (...,)
        Ellipticity angle [-pi/4, pi/4].
    docp : np.ndarray, (...,), optional
        Degree of Circular Polarization (DoCP) [0.0, 1.0].

    Returns
    -------
    cop_colored : np.ndarray, (..., 3), uint8
        Colored array of CoP.
    """
    return _colorize_cop_spec(eang, docp)[0]


def _colorize_stokes_spec(stokes: npt.NDArray, gamma: float = 1.0, halfrange: float | None = None) -> tuple[npt.NDArray[np.uint8], ColorizerSpec]:
    return _colorize_spec(stokes, cmap="stokes", norm=SymPowerNorm(gamma=gamma, halfrange=halfrange))


def colorize_stokes(stokes: npt.NDArray, gamma: float = 1.0, halfrange: float | None = None) -> npt.NDArray[np.uint8]:
    """Apply colormap to Stokes vector.

    Parameters
    ----------
    stokes : np.ndarray, (...,)
        Stokes vector image.
    gamma : float, optional
        Gamma for SymPowerNorm. Default: 1.0.
    halfrange : float, optional
        Half of the symmetric data range for the colormap. Actual limits are vmin=-halfrange and vmax=halfrange.

    Returns
    -------
    stokes_colored : np.ndarray, (..., 3), uint8
        Colored array of Stokes vector image.
    """
    return _colorize_stokes_spec(stokes, gamma=gamma, halfrange=halfrange)[0]


def _colorize_mueller_spec(mueller: npt.NDArray, gamma: float = 1.0, halfrange: float | None = None) -> tuple[npt.NDArray[np.uint8], ColorizerSpec]:
    return _colorize_spec(mueller, cmap="mueller", norm=SymPowerNorm(gamma=gamma, halfrange=halfrange))


def colorize_mueller(mueller: npt.NDArray, gamma: float = 1.0, halfrange: float | None = None) -> npt.NDArray[np.uint8]:
    """Apply colormap to Mueller matrix.

    Parameters
    ----------
    mueller : np.ndarray, (..., 4, 4)
        Mueller matrix.
    gamma : float, optional
        Gamma for SymPowerNorm. Default: 1.0.
    halfrange : float, optional
        Half of the symmetric data range for the colormap. Actual limits are vmin=-halfrange and vmax=halfrange.

    Returns
    -------
    mueller_colored : np.ndarray, (..., 3), uint8
        Colored array of Mueller matrix image.
    """
    return _colorize_mueller_spec(mueller, gamma=gamma, halfrange=halfrange)[0]
