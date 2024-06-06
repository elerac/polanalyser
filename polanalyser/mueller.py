import numpy as np
import numpy.typing as npt


def calcMueller(intensities: npt.ArrayLike, mm_psg: npt.ArrayLike, mm_psa: npt.ArrayLike) -> np.ndarray:
    """Calculate Mueller matrix from measured intensities and Mueller matrices of Polarization State Generator (PSG) and Polarization State Analyzer (PSA)

    This function calculates Mueller matrix image from intensity images captured under a variety of polarimetric conditions (both PSG and PSA).
    Polarimetric conditions are specified by the Mueller matrices (`mm_psg` and `mm_psa`).

    The unknown Mueller matrix is calculated by the least-squares method from pairs of intensities and Muller matrices.
    The number of input pairs must be greater than the number of Mueller matrix parameters (i.e., more than 9 or 16).

    Parameters
    ----------
    intensities : ArrayLike
        Intensities (N, *), where N is the number of intensities and * is the shape of the image (or tensor).
    mm_psg : ArrayLike
        Mueller matrix of the Polarization State Generator (PSG). (N, 3, 3) or (N, 4, 4)
    mm_psa : ArrayLike
        Mueller matrix of the Polarization State Analyzer (PSA). (N, 3, 3) or (N, 4, 4)

    Returns
    -------
    mueller : np.ndarray
        Mueller matrix. The last two channels are correspondence to Mueller matrix. (*, 3, 3) or (*, 4, 4)

    Examples
    --------
    >>> mueller_obj = 2 * np.random.rand(4, 4) - 1  # Unknown mueller matrix of target object
    >>> # mueller_obj = 2 * np.random.rand(128, 256, 4, 4) - 1  # You can expand to array (like image)
    >>> intensity_list = []
    >>> mm_psg_list = []
    >>> mm_psa_list = []
    >>> for angle in np.linspace(0, np.pi, num=36, endpoint=False):
    ...     mm_psg = pa.qwp(5 * angle) @ pa.polarizer(0)
    ...     mm_psa = pa.polarizer(np.pi / 2) @ pa.qwp(angle)
    ...     intensity = (mm_psa @ mueller_obj @ mm_psg)[..., 0, 0]
    ...     intensity_list.append(intensity)
    ...     mm_psg_list.append(mueller_psg)
    ...     mm_psa_list.append(mueller_psa)
    >>> mueller_pred = pa.calcMueller(intensity_list, mm_psg_list, mm_psa_list)
    >>> mueller_pred.shape
    (4, 4)
    >>> np.allclose(mueller_obj, mueller_pred)
    True
    """
    # Convert ArrayLike object to np.ndarray
    intensities = np.array(intensities)  # (N, *)
    mm_psa = np.array(mm_psa)  # (N, 3, 3) or (N, 4, 4) or (N, 3) or (N, 4)
    mm_psg = np.array(mm_psg)  # (N, 3, 3) or (N, 4, 4) or (N, 3) or (N, 4)
    num = len(intensities)

    # In case of stokes vector, expand the axis of the number of elements
    if mm_psg.ndim == 2:  # (N, 3) or (N, 4) -> (N, 1, 3) or (N, 1, 4)
        mm_psg = np.expand_dims(mm_psg, axis=1)

    if mm_psa.ndim == 2:  # (N, 3) or (N, 4) -> (N, 3, 1) or (N, 4, 1)
        mm_psa = np.expand_dims(mm_psa, axis=2)

    # Check the number of the input elements
    if not (len(intensities) == len(mm_psa) == len(mm_psg)):
        raise ValueError(f"The number of elements must be same. {len(intensities)} != {len(mm_psa)} != {len(mm_psg)}")

    # Check the shape of the Mueller matrices
    if not (mm_psg.ndim == 3 and mm_psg.ndim == 3):
        raise ValueError(f"The shape of mueller matrices must be (N, 3, 3) or (N, 4, 4), not {mm_psg.shape}, {mm_psa.shape}")

    m_h = mm_psg.shape[2]
    m_w = mm_psa.shape[1]

    # Construct the observation matrix
    W = np.empty((num, m_h * m_w))
    for i in range(num):
        s_psg = mm_psg[i, :, 0][:, np.newaxis]  # [m00, m10, m20] or [m00, m10, m20, m30] (3, 1) or (4, 1)
        s_psa = mm_psa[i, 0, :][np.newaxis, :]  # [m00, m01, m02] or [m00, m01, m02, m03] (1, 3) or (1, 4)
        W[i] = np.ravel((s_psg @ s_psa).T)
    W_pinv = np.linalg.pinv(W)

    intensities = np.moveaxis(intensities, 0, -1)  # (*, N)

    # Least-squares
    mueller = np.tensordot(W_pinv, intensities, axes=(-1, -1))  # (9, *) or (16, *)
    mueller = np.moveaxis(mueller, 0, -1)  # (*, 9) or (*, 16)
    mueller = np.reshape(mueller, (*mueller.shape[:-1], m_h, m_w))  # (*, 3, 3) or (*, 4, 4)
    return mueller


def rotator(theta: float) -> np.ndarray:
    """Generate Mueller matrix of rotation

    Parameters
    ----------
    theta : float
        The angle of rotation

    Returns
    -------
    mueller : np.ndarray
        Mueller matrix (4, 4)
    """
    s = np.sin(2 * theta)
    c = np.cos(2 * theta)
    return np.array([[1, 0, 0, 0], [0, c, s, 0], [0, -s, c, 0], [0, 0, 0, 1]])


def rotateMueller(mueller: np.ndarray, theta: float) -> np.ndarray:
    """Rotate Mueller matrix

    Parameters
    ----------
    mueller : np.ndarray
        Mueller matrix to rotate, (3, 3) or (4, 4)
    theta : float
        The angle of rotation

    Returns
    -------
    mueller_rotated : np.ndarray
        Rotated mueller matrix (3, 3) or (4, 4)
    """
    mueller_shape = mueller.shape[-2:]
    if mueller_shape == (4, 4):
        return rotator(-theta) @ mueller @ rotator(theta)
    elif mueller_shape == (3, 3):
        return rotator(-theta)[:3, :3] @ mueller @ rotator(theta)[:3, :3]
    else:
        raise ValueError(f"The shape of mueller matrix must be (3, 3) or (4, 4), not {mueller_shape}")


def polarizer(theta: float) -> np.ndarray:
    """Generate Mueller matrix of the linear polarizer

    Parameters
    ----------
    theta : float
        Angle of polarizer

    Returns
    -------
    mueller : np.ndarray
        Mueller matrix, (4, 4)
    """
    s = np.sin(2 * theta)
    c = np.cos(2 * theta)
    return 0.5 * np.array([[1, c, s, 0], [c, c * c, s * c, 0], [s, s * c, s * s, 0], [0, 0, 0, 0]])


def retarder(delta: float, theta: float) -> np.ndarray:
    """Generate Mueller matrix of linear retarder

    Parameters
    ----------
    delta : float
        Phase difference between the fast and slow axis
    theta : float
        Angle of the fast axis

    Returns
    -------
    mueller : np.ndarray
        Mueller matrix, (4, 4)
    """
    s = np.sin(delta)
    c = np.cos(delta)
    mueller = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, c, s], [0, 0, -s, c]])
    mueller = rotateMueller(mueller, theta)
    return mueller


def qwp(theta: float) -> np.ndarray:
    """Generate Mueller matrix of Quarter-Wave Plate (QWP)

    Parameters
    ----------
    theta : float
        Angle of the fast axis

    Returns
    -------
    mueller : np.ndarray
        Mueller matrix (4, 4)
    """
    return retarder(np.pi / 2, theta)


def hwp(theta: float) -> np.ndarray:
    """Generate Mueller matrix of Half-Wave Plate (HWP)

    Parameters
    ----------
    theta : float
        Angle of the fast axis

    Returns
    -------
    mueller : np.ndarray
        Mueller matrix (4, 4)
    """
    return retarder(np.pi, theta)
