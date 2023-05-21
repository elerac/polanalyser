from typing import List
import numpy as np
from numpy import diag
from numpy import dot
from numpy import zeros
from scipy.linalg import svd

# SVD pseudo inverse, sets all diagonals to zero except first 9 or 16 elements for a 3x3 or 4x4 matrix, respectively
def svdpinv(I: List[np.ndarray], PSA: List[np.ndarray]) -> np.ndarray::
	# svd, s is reciprocated and V is transposed
	UT, s, V = svd(I, full_matrices=False)

	# get number of singular values from PSA shape
	k = len(PSA[0,0])**2
	
	# set all diagonal values to zero except first k elements
	s[k:] = 0

	# get the diagonal matrix for s
	S = diag(s)

	# return the reconstructed matrix
	return UT.dot(S.dot(V))

# calculate Mueller calibration matrix W
def calcW(I: List[np.ndarray], PSA: List[np.ndarray], PSG: List[np.ndarray]) -> np.ndarray:
	# Check the shape of the input mueller matrices I = [len, xpix, ypix, n], W = [4, n]
	if len(I[0,0,0]) != len(W[0]):
		raise ValueError(f"The values of n must be equal, I = {len(I[0,0,0])} W = {len(W[0])}")
	length = len(I)
	psa_shape = PSA[0].shape
	I = np.moveaxis(I, 0, -1)  # (*, len)
	PSA = np.moveaxis(PSA, 0, -1)  # (*, len)
	PSG = np.moveaxis(PSG, 0, -1)  # (*, len)
	
	# combine PSA and PSG
	PSB = np.empty((length, np.prod(psa_shape)))
	for i in range(length):
		P1 = np.expand_dims(PSG[:, 0, i], axis=1)  # [m00, m10, m20] or [m00, m10, m20, m30]
		A1 = np.expand_dims(PSA[0, :, i], axis=0)  # [m00, m01, m02] or [m00, m01, m02, m03]
		PSB[i] = np.ravel((P1 @ A1).T)
	
	Ihat = svdpinv(I)
	return np.tensordot(PSB, Ihat, axes=(1, -1))

# calculate the Meuller matrix
def calcM(I: List[np.ndarray], W: List[np.ndarray], PSA: List[np.ndarray]) -> np.ndarray:
	psa_shape = PSA[0].shape
	I = np.moveaxis(I, 0, -1)  # (*, len)
	M = np.tensordot(W, I, axes=(1, -1))
	M = np.moveaxis(M, 0, -1)  # (*, 9) or (*, 16)
	M = np.reshape(M, (*M.shape[:-1], *psa_shape))  # (*, 3, 3) or (*, 4, 4)
	return M

def calcMueller(intensity_list: List[np.ndarray], mueller_psg_list: List[np.ndarray], mueller_psa_list: List[np.ndarray]) -> np.ndarray:
    """Calculate Mueller matrix from measured intensities and Mueller matrices of Polarization State Generator (PSG) and Polarization State Analyzer (PSA)

    This function calculates Mueller matrix image from intensity images captured under a variety of polarimetric conditions (both PSG and PSA).
    Polarimetric conditions are described by Mueller matrix form (`mueller_psg_list` and `mueller_psa_list`).

    The unknown Mueller matrix is calculated by the least-squares method from pairs of intensities and Muller matrices.
    The number of input pairs must be greater than the number of Mueller matrix parameters (i.e., more than 9 or 16).

    Parameters
    ----------
    intensity_list : List[np.ndarray]
        List of intensity
    mueller_psg_list : List[np.ndarray]
        List of mueller matrix of the Polarization State Generator (PSG). (3, 3) or (4, 4)
    mueller_psa_list : List[np.ndarray]
        List of mueller matrix of the Polarization State Analyzer (PSA). (3, 3) or (4, 4)

    Returns
    -------
    mueller : np.ndarray
        Mueller matrix. The last two channels are correspondence to Mueller matrix. (*, 3, 3) or (*, 4, 4)

    Examples
    --------
    >>> mueller_obj = 2 * np.random.rand(4, 4) - 1  # Unknown mueller matrix of target object
    >>> # mueller_obj = 2 * np.random.rand(128, 256, 4, 4) - 1  # You can expand to array (like image)
    >>> intensity_list = []
    >>> mueller_psg_list = []
    >>> mueller_psa_list = []
    >>> for angle in np.linspace(0, np.pi, num=36, endpoint=False):
    ...     mueller_psg = pa.qwp(5 * angle) @ pa.polarizer(0)
    ...     mueller_psa = pa.polarizer(np.pi / 2) @ pa.qwp(angle)
    ...     intensity = (mueller_psa @ mueller_obj @ mueller_psg)[..., 0, 0]
    ...     intensity_list.append(intensity)
    ...     mueller_psg_list.append(mueller_psg)
    ...     mueller_psa_list.append(mueller_psa)
    >>> mueller_pred = pa.calcMueller(intensity_list, mueller_psg_list, mueller_psa_list)
    >>> mueller_pred.shape
    (4, 4)
    >>> np.allclose(mueller_obj, mueller_pred)
    True
    """
    # Convert ArrayLike object to np.ndarray
    intensities = np.array(intensity_list)  # (len, *)
    muellers_psa = np.array(mueller_psa_list)  # (len, 3, 3) or (len, 4, 4)
    muellers_psg = np.array(mueller_psg_list)  # (len, 3, 3) or (len, 4, 4)

    # Check the number of the input elements
    len_intensities = len(intensities)
    len_muellers_psa = len(muellers_psa)
    len_muellers_psg = len(muellers_psg)
    if not (len_intensities == len_muellers_psa == len_muellers_psg):
        raise ValueError(f"The number of elements must be same. {len_intensities}, {len_muellers_psa}, {len_muellers_psg}.")

    # Check the shape of the input mueller matrices
    mueller_psg_shape = muellers_psg[0].shape
    mueller_psa_shape = muellers_psa[0].shape
    if mueller_psg_shape != mueller_psa_shape:
        raise ValueError(f"The shape of mueller matrices must be same, not {mueller_psg_shape} != {mueller_psa_shape}")

    if not (mueller_psg_shape == (3, 3) or mueller_psg_shape == (4, 4)):
        raise ValueError(f"The shape of mueller matrix must (3, 3) or (4, 4), not {mueller_psg_shape}")

    # Move the axis of the number of elements to the last axis
    intensities = np.moveaxis(intensities, 0, -1)  # (*, len)
    muellers_psa = np.moveaxis(muellers_psa, 0, -1)  # (*, len)
    muellers_psg = np.moveaxis(muellers_psg, 0, -1)  # (*, len)

    # Calculate
    length = len_intensities
    W = np.empty((length, np.prod(mueller_psa_shape)))
    for i in range(length):
        P1 = np.expand_dims(muellers_psg[:, 0, i], axis=1)  # [m00, m10, m20] or [m00, m10, m20, m30]
        A1 = np.expand_dims(muellers_psa[0, :, i], axis=0)  # [m00, m01, m02] or [m00, m01, m02, m03]
        W[i] = np.ravel((P1 @ A1).T)

    W_pinv = np.linalg.pinv(W)
    mueller = np.tensordot(W_pinv, intensities, axes=(1, -1))  # (9, *) or (16, *)
    mueller = np.moveaxis(mueller, 0, -1)  # (*, 9) or (*, 16)
    mueller = np.reshape(mueller, (*mueller.shape[:-1], *mueller_psa_shape))  # (*, 3, 3) or (*, 4, 4)
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
    mueller = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, c, -s], [0, 0, s, c]])
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
