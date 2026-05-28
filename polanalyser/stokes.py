"""Stokes vector related functions."""

import numpy as np
import numpy.typing as npt

from . import mueller


def stokes(s0: npt.ArrayLike, dop: npt.ArrayLike, aolp: npt.ArrayLike, eang: npt.ArrayLike) -> np.ndarray:
    """Generate Stokes vector from [s0, dop, aolp, eang].

    Parameters
    ----------
    s0 : array_like, (...,)
        Intensity of the light. Must be non-negative.
    dop : array_like, (...,)
        Degree of polarization in [0, 1].
    aolp : array_like, (...,)
        Angle of linear polarization in [0, pi].
    eang : array_like, (...,)
        Ellipticity angle in [-pi/4, pi/4].

    Returns
    -------
    stokes : ndarray, (..., 4)
        Stokes vector constructed from the inputs.

    Examples
    --------
    >>> pa.stokes(s0=1.0, dop=1.0, aolp=0.0, eang=0.0)  # Linear horizontal polarization
    [1. 1. 0. 0.]
    >>> pa.stokes(s0=1.0, dop=1.0, aolp=np.pi/4, eang=0.0)  # Linear +45 degree polarization
    [1. 0. 1. 0.]
    >>> pa.stokes(s0=1.0, dop=1.0, aolp=0.0, eang=np.pi/4)  # Right circular polarization
    [1. 0. 0. 1.]
    >>> pa.stokes(s0=1.0, dop=1.0, aolp=0.0, eang=-np.pi/4)  # Left circular polarization
    [ 1.  0.  0. -1.]
    >>> pa.stokes(s0=1.0, dop=0.0, aolp=0.0, eang=0.0)  # Unpolarized light
    [1. 0. 0. 0.]
    """
    s0 = np.asarray(s0)
    dop = np.asarray(dop)
    aolp = np.asarray(aolp)
    eang = np.asarray(eang)

    if np.any(s0 < 0):
        raise ValueError("Intensity must be non-negative")

    if np.any(np.logical_or(dop < 0, 1 < dop)):
        raise ValueError("Degree of polarization (dop) must be in the range [0, 1]")

    if np.any(np.logical_or(aolp < 0, np.pi < aolp)):
        raise ValueError("Angle of linear polarization (aolp) must be in the range [0, pi]")

    if np.any(np.logical_or(eang < -np.pi / 4, np.pi / 4 < eang)):
        raise ValueError("Ellipticity angle (eang) must be in the range [-pi/4, pi/4]")

    s0, dop, aolp, eang = np.broadcast_arrays(s0, dop, aolp, eang)
    s1 = s0 * dop * np.cos(2 * aolp) * np.cos(2 * eang)
    s2 = s0 * dop * np.sin(2 * aolp) * np.cos(2 * eang)
    s3 = s0 * dop * np.sin(2 * eang)
    return np.stack([s0, s1, s2, s3], axis=-1)


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
    stokes : ndarray
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
    # Convert ArrayLike object to ndarray
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
    stokes : ndarray
        Calculated stokes parameters
    """
    muellers = [mueller.polarizer(angle)[:3, :3] for angle in polarizer_angles]
    return calcStokes(intensities, muellers)


def _movelastaxis(a: npt.ArrayLike, source: int) -> np.ndarray:
    """Equivalent to `np.moveaxis(a, source, -1)` but does not move the axis if source is -1"""
    a = np.asarray(a)
    if source != -1:
        a = np.moveaxis(a, source, -1)
    return a


def stokes_to_dolp(stokes: npt.ArrayLike, axis: int = -1) -> np.ndarray:
    """Convert Stokes vector to DoLP (Degree of Linear Polarization).

    .. math::
        \\text{DoLP} = \\frac{\\sqrt{s_1^2 + s_2^2}}{s_0}.

    Parameters
    ----------
    stokes : array_like, (..., 3) or (..., 4)
        Stokes vector.
    axis : int, optional
        Axis of the stokes channel, by default -1.

    Returns
    -------
    dolp : ndarray, (...)
        DoLP [0, 1].

    Examples
    --------
    >>> pa.stokes_to_dolp([1.0, 0.0, 0.0, 0.0])  # Unpolarized light
    0.0
    >>> pa.stokes_to_dolp([1.0, 1.0, 0.0, 0.0])  # Fully linear polarized light
    1.0
    >>> pa.stokes_to_dolp([1.0, 0.0, 0.0, 1.0])  # Fully circularly polarized light
    0.0
    >>> pa.stokes_to_dolp([1.0, 0.5, 0.5, 0.5])  # Partially polarized light
    0.707
    """
    stokes = _movelastaxis(stokes, axis)
    s0 = stokes[..., 0]
    s1 = stokes[..., 1]
    s2 = stokes[..., 2]
    return np.hypot(s1, s2) / s0


def stokes_to_aolp(stokes: npt.ArrayLike, axis: int = -1) -> np.ndarray:
    """Convert Stokes vector to AoLP (Angle of Linear Polarization).

    .. math::
        \\text{AoLP} = \\frac{1}{2} \\tan^{-1} \\left( \\frac{s_2}{s_1} \\right).

    Parameters
    ----------
    stokes : array_like, (..., 3) or (..., 4)
        Stokes vector.
    axis : int, optional
        Axis of the stokes channel, by default -1.

    Returns
    -------
    aolp : ndarray, (...)
        AoLP [0, pi].

    Examples
    --------
    >>> pa.stokes_to_aolp([1.0, 1.0, 0.0, 0.0])  # Linear horizontal polarization
    0.0
    >>> pa.stokes_to_aolp([1.0, 0.0, 1.0, 0.0])  # Linear +45 degree polarization
    0.785
    >>> pa.stokes_to_aolp([1.0, -1.0, 0.0, 0.0])  # Linear vertical polarization
    1.571
    >>> pa.stokes_to_aolp([1.0, 0.0, -1.0, 0.0])  # Linear -45 degree polarization
    2.356
    """
    stokes = _movelastaxis(stokes, axis)
    s1 = stokes[..., 1]
    s2 = stokes[..., 2]
    aolp = 0.5 * np.arctan2(s2, s1)  # [-pi/2, pi/2]
    aolp = np.where(aolp < 0, aolp + np.pi, aolp)  # Map to [0, pi]
    return aolp


def stokes_to_dop(stokes: npt.ArrayLike, axis: int = -1) -> np.ndarray:
    """Convert Stokes vector to DoP (Degree of Polarization).

    .. math::
        \\text{DoP} = \\frac{\\sqrt{s_1^2 + s_2^2 + s_3^2}}{s_0}.

    Parameters
    ----------
    stokes : array_like, (..., 4)
        Stokes vector.
    axis : int, optional
        Axis of the stokes channel, by default -1.

    Returns
    -------
    dop : ndarray, (...)
        DoP [0, 1]

    Examples
    --------
    >>> pa.stokes_to_dop([1.0, 0.0, 0.0, 0.0])  # Unpolarized light
    0.0
    >>> pa.stokes_to_dop([1.0, 1.0, 0.0, 0.0])  # Fully linear polarized light
    1.0
    >>> pa.stokes_to_dop([1.0, 0.0, 0.0, 1.0])  # Fully circularly polarized light
    1.0
    >>> pa.stokes_to_dop([1.0, 0.5, 0.5, 0.5])  # Partially polarized light
    0.866
    """
    stokes = _movelastaxis(stokes, axis)
    s0 = stokes[..., 0]
    s1 = stokes[..., 1]
    s2 = stokes[..., 2]
    s3 = stokes[..., 3]
    return np.sqrt(s1**2 + s2**2 + s3**2) / s0


def stokes_to_eang(stokes: npt.ArrayLike, axis: int = -1) -> np.ndarray:
    """Convert Stokes vector to ellipticity angle.

    .. math::
        \\text{Ellipticity Angle} = \\frac{1}{2} \\tan^{-1} \\left( \\frac{s_3}{\\sqrt{s_1^2 + s_2^2}} \\right).

    Parameters
    ----------
    stokes : array_like, (..., 4)
        Stokes vector.
    axis : int, optional
        Axis of the stokes channel, by default -1.

    Returns
    -------
    enag : ndarray, (...)
        ellipticity angle [-pi/4, pi/4].

    Examples
    --------
    >>> pa.stokes_to_eang([1.0, 1.0, 0.0, 0.0])  # Linear horizontal polarization
    0.0
    >>> pa.stokes_to_eang([1.0, 0.0, 1.0, 0.0])  # Linear +45 degree polarization
    0.0
    >>> pa.stokes_to_eang([1.0, 0.0, 0.0, 1.0])  # Right circular polarization
    0.785
    >>> pa.stokes_to_eang([1.0, 0.0, 0.0, -1.0])  # Left circular polarization
    -0.785
    """
    stokes = _movelastaxis(stokes, axis)
    s1 = stokes[..., 1]
    s2 = stokes[..., 2]
    s3 = stokes[..., 3]
    return 0.5 * np.arctan2(s3, np.hypot(s1, s2))


def stokes_to_docp(stokes: npt.ArrayLike, axis: int = -1) -> np.ndarray:
    """Convert Stokes vector to DoCP (Degree of Circular Polarization).

    .. math::
        \\text{DoCP} = \\frac{|s_3|}{s_0}.

    Parameters
    ----------
    stokes : array_like, (..., 4)
        Stokes vector.
    axis : int, optional
        Axis of the stokes channel, by default -1.

    Returns
    -------
    docp : ndarray, (...)
        DoCP [0, 1].

    Examples
    --------
    >>> pa.stokes_to_docp([1.0, 0.0, 0.0, 0.0])  # Unpolarized light
    0.0
    >>> pa.stokes_to_docp([1.0, 1.0, 0.0, 0.0])  # Fully linear polarized light
    0.0
    >>> pa.stokes_to_docp([1.0, 0.0, 0.0, 1.0])  # Fully circularly polarized light
    1.0
    >>> pa.stokes_to_docp([1.0, 0.5, 0.5, 0.5])  # Partially polarized light
    0.5
    """
    stokes = _movelastaxis(stokes, axis)
    s0 = stokes[..., 0]
    s3 = stokes[..., 3]
    return np.abs(s3) / s0


def isstokes(stokes: npt.ArrayLike, *, rtol: float = 1.0e-5, atol: float = 0.0, axis: int = -1) -> np.ndarray:
    """Check if Stokes vector is physically valid.

    1. The intensity should be non-negative: :math:`s_0 \\geq 0`.
    2. The DoP should be smaller than or equal to 1: :math:`\\sqrt{s_1^2 + s_2^2 + s_3^2} \\leq s_0`.

    Parameters
    ----------
    stokes : array_like, (..., 3) or (..., 4)
        Stokes vector.
    rtol : float, optional
        Relative tolerance for the polarization boundary, by default 1.0e-5.
    atol : float, optional
        Absolute tolerance for the polarization boundary, by default 0.0.
    axis : int, optional
        The axis that contains the Stokes vectors, by default -1.

    Returns
    -------
    is_valid : ndarray, (..., )
        This is scalar if the input is a single Stokes vector, and an array of booleans if the input is a stack of Stokes vectors.

    Examples
    --------
    >>> pa.isstokes([1.0, 0.0, 0.0, 0.0])
    True
    >>> pa.isstokes([1.0, 1.0, 0.0, 0.0])
    True
    >>> pa.isstokes([1.0, 1.01, 0.0, 0.0])
    False
    >>> pa.isstokes([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.01, 0.0, 0.0]])
    [ True  True False]
    """
    if rtol < 0:
        raise ValueError("Relative tolerance (rtol) must be non-negative")
    if atol < 0:
        raise ValueError("Absolute tolerance (atol) must be non-negative")

    stokes = _movelastaxis(stokes, axis)
    if stokes.shape[-1] not in (3, 4):
        raise ValueError(f"Invalid shape: {stokes.shape}. Expected a Stokes axis with 3 or 4 components.")

    s0 = stokes[..., 0]
    p = np.linalg.norm(stokes[..., 1:], axis=-1)

    # The intensity should be non-negative
    # s0 >= 0
    is_valid_intensity = s0 >= 0

    # The polarization magnitude should not exceed the intensity, except for
    # numerical boundary error scaled in intensity units.
    is_valid_polarization = p <= s0 + atol + rtol * np.abs(s0)

    # NaN and inf cannot represent physical Stokes vectors.
    is_finite = np.all(np.isfinite(stokes), axis=-1)

    return is_finite & is_valid_intensity & is_valid_polarization
