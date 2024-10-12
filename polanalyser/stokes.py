import numpy as np
import numpy.typing as npt
from .mueller import polarizer


def calcStokes(intensities: npt.ArrayLike, muellers: npt.ArrayLike) -> np.ndarray:
    """Calculate stokes parameters from measured intensities and mueller matrices

    Parameters
    ----------
    intensity_list : ArrayLike
        Intensities (N, *)
    mueller_list : ArrayLike
        Mueller matrices (N, 3, 3) or (N, 4, 4), or Stokes vectors (N, 3) or (N, 4). If the shape is (N,), this function treats as the angles of linear polarizer.

    Returns
    -------
    stokes : np.ndarray
        Calculated stokes parameters

    Examples
    --------
    Calculate the unknown stokes parameters from the measured intensity with a rotating polarizer

    >>> stokes = np.array([1.0, 0.1, -0.3])  # Unknown stokes parameters (without circular polarization)
    >>> intensity_list = []
    >>> mueller_list = []
    >>> for angle in np.deg2rad([0, 45, 90, 135]):
    ...     mueller = pa.polarizer(angle)[:3, :3]
    ...     intensity = (mueller @ stokes)[0]
    ...     intensity_list.append(intensity)
    ...     mueller_list.append(mueller)
    >>> stokes_pred = pa.calcStokes(intensity_list, mueller_list)
    >>> stokes_pred
    [1.0, 0.1, -0.3]
    >>> np.allclose(stokes, stokes_pred)
    True

    Calculate the unknown stokes parameters from the measured intensity with QWP and polarizer

    >>> stokes = np.array([1.0, 0.1, -0.3, 0.01])  # Unknown stokes parameters
    >>> intensity_list = []
    >>> mueller_list = []
    >>> for angle in np.deg2rad([0.0, 22.5, 45.0, 67.5]):
    ...     mueller = pa.polarizer(0) @ pa.qwp(angle)
    ...     intensity = (mueller @ stokes)[0]
    ...     intensity_list.append(intensity)
    ...     mueller_list.append(mueller)
    >>> stokes_pred = pa.calcStokes(intensity_list, mueller_list)
    >>> stokes_pred
    [1.0, 0.1, -0.3, 0.01]
    >>> np.allclose(stokes, stokes_pred)
    True
    """
    # Convert ArrayLike object to np.ndarray
    intensities = np.array(intensities)  # (N, *)
    muellers = np.array(muellers)  # (N, *)

    # If the shape of `muellers` is a 1D array (each element is scalar), this function treats `muellers` as the angles of a linear polarizer.
    if muellers.ndim == 1:
        polarizer_angles = muellers
        return calcLinearStokes(intensities, polarizer_angles)

    # Check the number of elements
    if len(intensities) != len(muellers):
        raise ValueError(f"The number of elements must be same, not {len(intensities)} != {len(muellers)}")

    # In case of stokes vector (N, 3) or (N, 4), expand the axis for later matrix manipulation
    if muellers.ndim == 2:  # (N, 3) or (N, 4) -> (N, 1, 3) or (N, 1, 4)
        muellers = muellers[:, np.newaxis, :]

    # Move the axis of the number of elements to the last axis
    intensities = np.moveaxis(intensities, 0, -1)  # (*, N)
    muellers = np.moveaxis(muellers, 0, -1)  # (*, N, 3) or (*, N, 4)

    # Calculate
    A = muellers[0].T  # [m00, m01, m02] (N, 3) or [m01, m02, m03, m04] (N, 4)
    A_pinv = np.linalg.pinv(A)  # (3, N) or (N, 4)
    stokes = np.tensordot(A_pinv, intensities, axes=(1, -1))  # (3, *) or (4, *)
    stokes = np.moveaxis(stokes, 0, -1)  # (*, 3) or (*, 4)
    return stokes


def calcLinearStokes(intensities: npt.ArrayLike, polarizer_angles: npt.ArrayLike) -> np.ndarray:
    """Calculate only linear polarization stokes parameters from measured intensities and linear polarizer angle

    Parameters
    ----------
    intensities : ArrayLike
        Intensities (N, *)
    angles : ArrayLike
        Polarizer angles (N,) in radian

    Returns
    -------
    stokes : np.ndarray
        Calculated stokes parameters
    """
    muellers = [polarizer(angle)[:3, :3] for angle in polarizer_angles]
    return calcStokes(intensities, muellers)


def _movelastaxis(a: np.ndarray, source: int) -> np.ndarray:
    """Equivalent to `np.moveaxis(a, source, -1)` but does not move the axis if source is -1"""
    if source != -1:
        a = np.moveaxis(a, source, -1)
    return a


def cvtStokesToImax(stokes: np.ndarray, axis: int = -1) -> np.ndarray:
    """Convert stokes parameters to Imax (maximum value when rotating the linear polarizer)

    Parameters
    ----------
    stokes : np.ndarray
        Stokes parameters
    axis : int, optional
        Axis of the stokes channel, by default -1

    Returns
    -------
    i_max : np.ndarray
        Imax
    """
    stokes = _movelastaxis(stokes, axis)
    s0 = stokes[..., 0]
    s1 = stokes[..., 1]
    s2 = stokes[..., 2]
    return (s0 + np.sqrt(s1**2 + s2**2)) * 0.5


def cvtStokesToImin(stokes: np.ndarray, axis: int = -1) -> np.ndarray:
    """Convert stokes parameters to Imin (minimum value when rotating the linear polarizer)

    Parameters
    ----------
    stokes : np.ndarray
        Stokes parameters
    axis : int, optional
        Axis of the stokes channel, by default -1

    Returns
    -------
    i_min : np.ndarray
        Imin
    """
    stokes = _movelastaxis(stokes, axis)
    s0 = stokes[..., 0]
    s1 = stokes[..., 1]
    s2 = stokes[..., 2]
    return (s0 - np.sqrt(s1**2 + s2**2)) * 0.5


def cvtStokesToDoLP(stokes: np.ndarray, axis: int = -1) -> np.ndarray:
    """Convert stokes parameters to DoLP (Degree of Linear Polarization)

    Parameters
    ----------
    stokes : np.ndarray
        Stokes parameters
    axis : int, optional
        Axis of the stokes channel, by default -1

    Returns
    -------
    DoLP : np.ndarray
        DoLP ∈ [0, 1]
    """
    stokes = _movelastaxis(stokes, axis)
    s0 = stokes[..., 0]
    s1 = stokes[..., 1]
    s2 = stokes[..., 2]
    return np.sqrt(s1**2 + s2**2) / s0


def cvtStokesToAoLP(stokes: np.ndarray, axis: int = -1) -> np.ndarray:
    """Convert stokes parameters to AoLP (Angle of Linear Polarization)

    Parameters
    ----------
    stokes : np.ndarray
        Stokes parameters
    axis : int, optional
        Axis of the stokes channel, by default -1

    Returns
    -------
    AoLP : np.ndarray
        AoLP ∈ [0, np.pi]
    """
    stokes = _movelastaxis(stokes, axis)
    s1 = stokes[..., 1]
    s2 = stokes[..., 2]
    return np.mod(0.5 * np.arctan2(s2, s1), np.pi)


def cvtStokesToIntensity(stokes: np.ndarray, axis: int = -1) -> np.ndarray:
    """Convert stokes parameters to intensity (same as s0 component)

    Parameters
    ----------
    stokes : np.ndarray
        Stokes parameters
    axis : int, optional
        Axis of the stokes channel, by default -1

    Returns
    -------
    intensity : np.ndarray
        Intensity
    """
    stokes = _movelastaxis(stokes, axis)
    s0 = stokes[..., 0]
    return s0


def cvtStokesToDiffuse(stokes: np.ndarray, axis: int = -1) -> np.ndarray:
    """Convert stokes parameters to diffuse

    Parameters
    ----------
    stokes : np.ndarray
        Stokes parameters
    axis : int, optional
        Axis of the stokes channel, by default -1

    Returns
    -------
    diffuse : np.ndarray
        Diffuse
    """
    Imin = cvtStokesToImin(stokes, axis)
    return Imin


def cvtStokesToSpecular(stokes: np.ndarray, axis: int = -1) -> np.ndarray:
    """Convert stokes parameters to specular

    Parameters
    ----------
    stokes : np.ndarray
        Stokes parameters
    axis : int, optional
        Axis of the stokes channel, by default -1

    Returns
    -------
    specular : np.ndarray
        Specular
    """
    stokes = _movelastaxis(stokes, axis)
    s1 = stokes[..., 1]
    s2 = stokes[..., 2]
    return np.sqrt(s1**2 + s2**2)  # same as Imax - Imin


def cvtStokesToDoP(stokes: np.ndarray, axis: int = -1) -> np.ndarray:
    """Convert stokes parameters to DoP (Degree of Polarization)

    Parameters
    ----------
    stokes : np.ndarray
        Stokes parameters
    axis : int, optional
        Axis of the stokes channel, by default -1

    Returns
    -------
    DoP : np.ndarray
        DoP ∈ [0, 1]
    """
    stokes = _movelastaxis(stokes, axis)
    s0 = stokes[..., 0]
    s1 = stokes[..., 1]
    s2 = stokes[..., 2]
    s3 = stokes[..., 3]
    return np.sqrt(s1**2 + s2**2 + s3**2) / s0


def cvtStokesToEllipticityAngle(stokes: np.ndarray, axis: int = -1) -> np.ndarray:
    """Convert stokes parameters to ellipticity angle

    Parameters
    ----------
    stokes : np.ndarray
        Stokes parameters
    axis : int, optional
        Axis of the stokes channel, by default -1

    Returns
    -------
    EllipticityAngle : np.ndarray
        ellipticity angle ∈ [-pi/4, pi/4]
    """
    stokes = _movelastaxis(stokes, axis)
    s1 = stokes[..., 1]
    s2 = stokes[..., 2]
    s3 = stokes[..., 3]
    return 0.5 * np.arctan2(s3, np.sqrt(s1**2 + s2**2))


def cvtStokesToDoCP(stokes: np.ndarray, axis: int = -1) -> np.ndarray:
    """Convert stokes parameters to DoCP (Degree of Circular Polarization)

    Parameters
    ----------
    stokes : np.ndarray
        Stokes parameters
    axis : int, optional
        Axis of the stokes channel, by default -1

    Returns
    -------
    DoCP : np.ndarray
        DoCP ∈ [0, 1]
    """
    stokes = _movelastaxis(stokes, axis)
    s0 = stokes[..., 0]
    s3 = stokes[..., 3]
    return np.abs(s3) / s0


def isstokes(stokes: npt.ArrayLike, atol: float = 1.0e-8, axis: int = -1) -> np.ndarray:
    """Check if the Stokes vector is physically valid.

    Parameters
    ----------
    stokes : (..., 4) array_like
        Stokes vector.
    atol : float, optional
        Absolute tolerance, by default 1.0e-8.
    axis : int, optional
        The axis that contains the Stokes vectors, by default -1.

    Returns
    -------
    is_valid : (..., ) array
        This is scalar if the input is a single Stokes vector, and an array of booleans if the input is a stack of Stokes vectors.

    Examples
    --------
    >>> pa.isstokes([1.0, 0.0, 0.0, 0.0])
    True
    >>> pa.isstokes([1.0, 1.0, 0.0, 0.0])
    True
    >>> pa.isstokes([1.0, 1.01, 0.0, 0.0])
    False
    """
    stokes = np.asarray(stokes)
    stokes = _movelastaxis(stokes, axis)
    s0 = stokes[..., 0]
    s1 = stokes[..., 1]
    s2 = stokes[..., 2]
    s3 = stokes[..., 3]

    # The intensity should be non-negative
    # s0 >= 0
    is_valid_intensity = s0 >= 0

    # The DoP should be smaller than 1
    # (s0**2 - (s1**2 + s2**2 + s3**2)) >= 0
    # but allow a small negative value due to numerical errors
    is_valid_dop = (s0**2 - (s1**2 + s2**2 + s3**2)) > -abs(atol)

    return np.bitwise_and(is_valid_intensity, is_valid_dop)
