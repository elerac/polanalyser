import numpy as np
import numpy.typing as npt

from .stokes import isstokes, stokes_to_dop


def stokes_to_jones(stokes: npt.ArrayLike) -> npt.NDArray[np.complex128]:
    """Convert Stokes vector to Jones vector.

    Parameters
    ----------
    stokes : array_like, (..., 4)
        Stokes vector with DoP = 1.0.

    Returns
    -------
    jones : np.ndarray, (..., 2), complex
        Jones vector.
    """
    stokes = np.asarray(stokes)

    if not np.all(isstokes(stokes)):
        raise ValueError("Invalid Stokes vector")

    if not np.allclose(stokes_to_dop(stokes), 1.0):
        raise ValueError("Stokes vector DoP must be 1.0")

    s0 = stokes[..., 0]
    s1 = stokes[..., 1]
    s2 = stokes[..., 2]
    s3 = stokes[..., 3]

    # Compute the magnitudes and relative phase (phi_x = 0)
    Ex = np.sqrt((s0 + s1) / 2.0)
    Ey = np.sqrt((s0 - s1) / 2.0)
    phi = np.arctan2(s3, s2)

    # Construct the Jones vector
    # Up to a global phase factor, we can choose Ex as real & positive
    Jx = Ex
    Jy = Ey * np.exp(1j * phi)
    jones = np.stack([Jx, Jy], axis=-1)
    return jones


def jones_to_stokes(jones: npt.ArrayLike) -> npt.NDArray[np.floating]:
    """Convert Jones vector to Stokes vector.

    Parameters
    ----------
    jones : array_like, (..., 2), complex
        Jones vector.

    Returns
    -------
    stokes : np.ndarray, (..., 4)
        Stokes vector.
    """
    jones = np.asarray(jones)
    Ex, Ey = jones[..., 0], jones[..., 1]
    s0 = np.abs(Ex) ** 2 + np.abs(Ey) ** 2
    s1 = np.abs(Ex) ** 2 - np.abs(Ey) ** 2
    s2 = 2 * np.real(Ex * np.conj(Ey))
    s3 = -2 * np.imag(Ex * np.conj(Ey))
    return np.stack([s0, s1, s2, s3], axis=-1)
