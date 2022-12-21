from typing import List, Optional
import numpy as np

def calcMueller(intensities: List[np.ndarray], muellers_psg: List[np.ndarray], muellers_psa: List[np.ndarray]):
    """Calculate Mueller matrix from observed intensities and Mueller matrixes of Polarization State Generator (PSG) and Polarization State Analyzer (PSA)

    This function calculates Mueller matrix image from images captured under a variety of polarimetric conditions (both PSG and PSA).
    Polarimetric conditions are described by Mueller matrix form (`muellers_psg` and `muellers_psa`).

    The unknown Mueller matrix is calculated by the least-squares method from pairs of intensities and Muller matrices.
    The number of input pairs must be greater than the number of Mueller matrix parameters (i.e., more than 9 or 16).
    
    Parameters
    ----------
    intensities : List[np.ndarray]
        Measured intensities.
    muellers_psg : List[np.ndarray]
        Mueller matrix of the Polarization State Generator (PSG). (3, 3) or (4, 4)
    muellers_psa : List[np.ndarray]
        Mueller matrix of the Polarization State Analyzer (PSA). (3, 3) or (4, 4)

    Returns
    -------
    mueller : np.ndarray
        Mueller matrix. (height, width, 9) or (height, width, 16)
    """
    lists_length = [len(intensities), len(muellers_psg), len(muellers_psa)]
    if not all(x == lists_length[0] for x in lists_length):
        raise ValueError(f"The length of the list must be the same, not {lists_length}.")

    # Convert List[np.ndarray] to np.ndarray
    intensities = np.stack(intensities, axis=-1)  # (height, width, K)
    muellers_psg = np.stack(muellers_psg, axis=-1)  # (3, 3, K) or (4, 4, K)
    muellers_psa = np.stack(muellers_psa, axis=-1)  # (3, 3, K) or (4, 4, K)
    
    K = intensities.shape[-1]  # scalar
    D = muellers_psg.shape[0]  # sclar: 3 or 4
    W = np.empty((K, D*D))
    for k in range(K):
        P1 = np.expand_dims(muellers_psg[:, 0, k], axis=1)  # [m00, m10, m20] or [m00, m10, m20, m30]
        A1 = np.expand_dims(muellers_psa[0, :, k], axis=0)  # [m00, m01, m02] or [m00, m01, m02, m03]
        W[k] = np.ravel((P1 @ A1).T)

    W_pinv = np.linalg.pinv(W)  # (D*D, K)
    mueller = np.tensordot(W_pinv, intensities, axes=(1, -1))  # (9, height, width) or (16, height, width)
    mueller = np.moveaxis(mueller, 0, -1)  # (height, width, 9) or ((height, width, 16)
    return mueller

def rotator(theta):
    """Generate Mueller matrix of rotation

    Parameters
    ----------
    theta : float
      the angle of rotation

    Returns
    -------
    mueller : np.ndarray
      mueller matrix (4, 4)
    """
    ones = np.ones_like(theta)
    zeros = np.zeros_like(theta)
    sin2 = np.sin(2*theta)
    cos2 = np.cos(2*theta)
    mueller = np.array([[ones,  zeros, zeros, zeros],
                  [zeros,  cos2,  sin2, zeros],
                  [zeros, -sin2,  cos2, zeros],
                  [zeros, zeros, zeros, ones]])
    mueller = np.moveaxis(mueller, [0,1], [-2,-1])
    return mueller

def rotateMueller(mueller, theta):
    """Rotate Mueller matrix
    
    Parameters
    ----------
    mueller : np.ndarray
      mueller matrix (3, 3) or (4, 4)
    theta : float
      the angle of rotation

    Returns
    -------
    mueller_rotated : np.ndarray
      rotated mueller matrix (3, 3) or (4, 4)

    Examples
    --------
    4x4 mueller matrix

    >>> M = np.random.rand(4, 4)
    >>> theta = np.pi/3  # 60 degree
    >>> M_rotated = rotateMueller(M, theta)

    3x3 mueller matrix

    >>> M3x3 = np.random.rand(3, 3)
    >>> M3x3_rotated = rotateMueller(M, np.pi/3)

    Tensor mueller matrix
    
    >>> img_mueller = np.random.rand(480, 640, 4, 4)
    >>> img_mueller_rotated = rotateMueller(img_mueller, np.pi/3)
    """
    if mueller.shape[-2:] == (4, 4):
        return rotator(-theta) @ mueller @ rotator(theta)
    elif mueller.shape[-2:] == (3, 3):
        return rotator(-theta)[:3, :3] @ mueller @ rotator(theta)[:3, :3]

def polarizer(theta):
    """Generate Mueller matrix of linear polarizer

    Parameters
    ----------
    theta : float
      the angle of the linear polarizer

    Returns
    -------
    mueller : np.ndarray
      mueller matrix (4, 4)
    """
    mueller = np.array([[0.5, 0.5, 0, 0],
                  [0.5, 0.5, 0, 0],
                  [  0,   0, 0, 0],
                  [  0,   0, 0, 0]]) # (4, 4)
    mueller = rotateMueller(mueller, theta)
    return mueller

def retarder(delta, theta):
    """Generate Mueller matrix of linear retarder
    
    Parameters
    ----------
    delta : float
      the phase difference between the fast and slow axis
    theta : float
      the angle of the fast axis

    Returns
    -------
    mueller : np.ndarray
      mueller matrix (4, 4)
    """
    ones = np.ones_like(delta)
    zeros = np.zeros_like(delta)
    sin = np.sin(delta)
    cos = np.cos(delta)
    mueller = np.array([[ones,  zeros, zeros, zeros],
                        [zeros, ones,  zeros, zeros],
                        [zeros, zeros, cos,   -sin],
                        [zeros, zeros, sin,   cos]])
    mueller = np.moveaxis(mueller, [0,1], [-2,-1])
    
    mueller = rotateMueller(mueller, theta)
    return mueller

def qwp(theta):
    """Generate Mueller matrix of Quarter-Wave Plate (QWP)
    
    Parameters
    ----------
    theta : float
      the angle of the fast axis

    Returns
    -------
    mueller : np.ndarray
      mueller matrix (4, 4)
    """
    return retarder(np.pi/2, theta)

def hwp(theta):
    """Generate Mueller matrix of Half-Wave Plate (QWP)
    
    Parameters
    ----------
    theta : float
      the angle of the fast axis

    Returns
    -------
    mueller : np.ndarray
      mueller matrix (4, 4)
    """
    return retarder(np.pi, theta)
