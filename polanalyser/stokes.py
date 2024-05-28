from typing import List
import warnings
import numpy as np
from .mueller import polarizer


def calcStokes(intensity_list: List[np.ndarray], mueller_list: List[np.ndarray]) -> np.ndarray:
    """Calculate stokes parameters from measured intensities and mueller matrices

    Parameters
    ----------
    intensity_list : List[np.ndarray]
        List of intensity
    mueller_list : List[np.ndarray]
        List of mueller matrix

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
    intensities = np.array(intensity_list)  # (len, *)
    muellers = np.array(mueller_list)  # (len, *)

    # Check the number of elements
    len_intensities = len(intensities)
    len_muellers = len(muellers)
    if len_intensities != len_muellers:
        raise ValueError(f"The number of elements must be same, not {len_intensities} != {len_muellers}.")

    # If the shape of `muellers` is a 1D array (each element is scalar), this function treats `muellers` as the angles of a linear polarizer.
    if muellers.ndim == 1:
        polarizer_angles = muellers
        return calcLinearStokes(intensities, polarizer_angles)

    # Move the axis of the number of elements to the last axis
    intensities = np.moveaxis(intensities, 0, -1)  # (*, len)
    muellers = np.moveaxis(muellers, 0, -1)  # (*, len)

    # Calculate
    A = muellers[0].T  # [m11, m12, m13] (len, 3) or [m11, m12, m13, m14] (len, 4)
    A_pinv = np.linalg.pinv(A)  # (3, len) or (len, 4)
    stokes = np.tensordot(A_pinv, intensities, axes=(1, -1))  # (3, *) or (4, *)
    stokes = np.moveaxis(stokes, 0, -1)  # (*, 3) or (*, 4)
    return stokes


def calcLinearStokes(intensities: List[np.ndarray], polarizer_angles: List[float]) -> np.ndarray:
    """Calculate only linear polarization stokes parameters from measured intensities and linear polarizer angle

    Parameters
    ----------
    intensities : List[np.ndarray]
        List of intensities (ors)
    muellers : List[np.ndarray]
        List of the angles of linear polarizer

    Returns
    -------
    stokes : np.ndarray
        Calculated stokes parameters
    """
    muellers = [polarizer(angle)[:3, :3] for angle in polarizer_angles]
    return calcStokes(intensities, muellers)


def cvtStokesToImax(stokes: np.ndarray) -> np.ndarray:
    """Convert stokes parameters to Imax (maximum value when rotating the linear polarizer)

    Parameters
    ----------
    stokes : np.ndarray
        Stokes parameters

    Returns
    -------
    i_max : np.ndarray
        Imax
    """
    s0 = stokes[..., 0]
    s1 = stokes[..., 1]
    s2 = stokes[..., 2]
    return (s0 + np.sqrt(s1**2 + s2**2)) * 0.5


def cvtStokesToImin(stokes: np.ndarray) -> np.ndarray:
    """Convert stokes parameters to Imin (minimum value when rotating the linear polarizer)

    Parameters
    ----------
    stokes : np.ndarray
        Stokes parameters

    Returns
    -------
    i_min : np.ndarray
        Imin
    """
    s0 = stokes[..., 0]
    s1 = stokes[..., 1]
    s2 = stokes[..., 2]
    return (s0 - np.sqrt(s1**2 + s2**2)) * 0.5


def cvtStokesToDoLP(stokes: np.ndarray) -> np.ndarray:
    """Convert stokes parameters to DoLP (Degree of Linear Polarization)

    Parameters
    ----------
    stokes : np.ndarray
        Stokes parameters

    Returns
    -------
    DoLP : np.ndarray
        DoLP ∈ [0, 1]
    """
    s0 = stokes[..., 0]
    s1 = stokes[..., 1]
    s2 = stokes[..., 2]
    return np.sqrt(s1**2 + s2**2) / s0


def cvtStokesToAoLP(stokes: np.ndarray) -> np.ndarray:
    """Convert stokes parameters to AoLP (Angle of Linear Polarization)

    Parameters
    ----------
    stokes : np.ndarray
        Stokes parameters

    Returns
    -------
    AoLP : np.ndarray
        AoLP ∈ [0, np.pi]
    """
    s1 = stokes[..., 1]
    s2 = stokes[..., 2]
    return np.mod(0.5 * np.arctan2(s2, s1), np.pi)


def cvtStokesToIntensity(stokes: np.ndarray) -> np.ndarray:
    """Convert stokes parameters to intensity (same as s0 component)

    Parameters
    ----------
    stokes : np.ndarray
        Stokes parameters

    Returns
    -------
    intensity : np.ndarray
        Intensity
    """
    s0 = stokes[..., 0]
    return s0


def cvtStokesToDiffuse(stokes: np.ndarray) -> np.ndarray:
    """Convert stokes parameters to diffuse

    Parameters
    ----------
    stokes : np.ndarray
        Stokes parameters
    Returns
    -------
    diffuse : np.ndarray
        Diffuse
    """
    Imin = cvtStokesToImin(stokes)
    return Imin


def cvtStokesToSpecular(stokes: np.ndarray) -> np.ndarray:
    """Convert stokes parameters to specular

    Parameters
    ----------
    stokes : np.ndarray
        Stokes parameters

    Returns
    -------
    specular : np.ndarray
        Specular
    """
    s1 = stokes[..., 1]
    s2 = stokes[..., 2]
    return np.sqrt(s1**2 + s2**2)  # same as Imax - Imin


def cvtStokesToDoP(stokes: np.ndarray) -> np.ndarray:
    """Convert stokes parameters to DoP (Degree of Polarization)

    Parameters
    ----------
    stokes : np.ndarray
        Stokes parameters

    Returns
    -------
    DoP : np.ndarray
        DoP ∈ [0, 1]
    """
    s0 = stokes[..., 0]
    s1 = stokes[..., 1]
    s2 = stokes[..., 2]
    s3 = stokes[..., 3]
    return np.sqrt(s1**2 + s2**2 + s3**2) / s0


def cvtStokesToEllipticityAngle(stokes: np.ndarray) -> np.ndarray:
    """Convert stokes parameters to ellipticity angle

    Parameters
    ----------
    stokes : np.ndarray
        Stokes parameters

    Returns
    -------
    EllipticityAngle : np.ndarray
        ellipticity angle ∈ [-pi/4, pi/4]
    """
    s1 = stokes[..., 1]
    s2 = stokes[..., 2]
    s3 = stokes[..., 3]
    return 0.5 * np.arctan2(s3, np.sqrt(s1**2 + s2**2))


def cvtStokesToDoCP(stokes: np.ndarray) -> np.ndarray:
    """Convert stokes parameters to DoCP (Degree of Circular Polarization)

    Parameters
    ----------
    stokes : np.ndarray
        Stokes parameters

    Returns
    -------
    DoCP : np.ndarray
        DoCP ∈ [-1, 1]
    """
    warnings.warn("The definition of the DoCP will be changed in the future update. If you want to use the new definition, please reinstall the polanalyser with the following command:\n $ pip install git+https://github.com/elerac/polanalyser.git@next\n", FutureWarning)

    s0 = stokes[..., 0]
    s3 = stokes[..., 3]
    return s3 / s0
