import cv2
import numpy as np
from .mueller import polarizer
from .util import njit_if_available

def calcStokes(intensities, muellers):
    """Calculate stokes vector from observed intensity and mueller matrix

    Parameters
    ----------
    intensities : np.ndarray
      Intensity of measurements (height, width, n)
    muellers : np.ndarray
      Mueller matrix (3, 3, n) or (4, 4, n)

    Returns
    -------
    stokes : np.ndarray
      Stokes vector (height, width, 3) or (height, width, 4)
    """
    if not isinstance(intensities, np.ndarray):
        intensities = np.stack(intensities, axis=-1) # (height, width, n)

    if not isinstance(muellers, np.ndarray):
        muellers = np.stack(muellers, axis=-1) # (3, 3, n) or (4, 4, n)

    if muellers.ndim == 1:
        # 1D array case
        thetas = muellers
        return calcLinearStokes(intensities, thetas)

    A = muellers[0].T # [m11, m12, m13] (n, 3) or [m11, m12, m13, m14] (n, 4)
    A_pinv = np.linalg.pinv(A) # (3, n)
    stokes = np.tensordot(A_pinv, intensities, axes=(1, -1)) # (3, height, width) or (4, height, width)
    stokes = np.moveaxis(stokes, 0, -1) # (height, width, 3)
    return stokes

def calcLinearStokes(intensities, thetas):
    """Calculate only linear polarization stokes vector from observed intensity and linear polarizer angle
    
    Parameters
    ----------
    intensities : np.ndarray
      Intensity of measurements (height, width, n)
    theta : np.ndarray
      Linear polarizer angles (n, )

    Returns
    -------
    S : np.ndarray
      Stokes vector (height, width, 3)
    """
    muellers = [ polarizer(theta)[..., :3, :3] for theta in thetas ]
    return calcStokes(intensities, muellers)

@njit_if_available(parallel=True, cache=True)
def cvtStokesToImax(img_stokes):
    """
    Convert stokes vector image to Imax image

    Parameters
    ----------
    img_stokes : np.ndarray, (height, width, 3)
        Stokes vector image

    Returns
    -------
    img_Imax : np.ndarray, (height, width)
        Imax image
    """
    S0 = img_stokes[..., 0]
    S1 = img_stokes[..., 1]
    S2 = img_stokes[..., 2]
    return (S0+np.sqrt(S1**2+S2**2))*0.5

@njit_if_available(parallel=True, cache=True)
def cvtStokesToImin(img_stokes):
    """
    Convert stokes vector image to Imin image

    Parameters
    ----------
    img_stokes : np.ndarray, (height, width, 3)
        Stokes vector image

    Returns
    -------
    img_Imin : np.ndarray, (height, width)
        Imin image
    """
    S0 = img_stokes[..., 0]
    S1 = img_stokes[..., 1]
    S2 = img_stokes[..., 2]
    return (S0-np.sqrt(S1**2+S2**2))*0.5

@njit_if_available(parallel=True, cache=True)
def cvtStokesToDoLP(img_stokes):
    """
    Convert stokes vector image to DoLP (Degree of Linear Polarization) image

    Parameters
    ----------
    img_stokes : np.ndarray, (height, width, 3)
        Stokes vector image

    Returns
    -------
    img_DoLP : np.ndarray, (height, width)
        DoLP image ∈ [0, 1]
    """
    S0 = img_stokes[..., 0]
    S1 = img_stokes[..., 1]
    S2 = img_stokes[..., 2]
    return np.sqrt(S1**2+S2**2)/S0

@njit_if_available(parallel=True, cache=True)
def cvtStokesToAoLP(img_stokes):
    """
    Convert stokes vector image to AoLP (Angle of Linear Polarization) image

    Parameters
    ----------
    img_stokes : np.ndarray, (height, width, 3)
        Stokes vector image

    Returns
    -------
    img_AoLP : np.ndarray, (height, width)
        AoLP image ∈ [0, np.pi]
    """
    S1 = img_stokes[..., 1]
    S2 = img_stokes[..., 2]
    return np.mod(0.5*np.arctan2(S2, S1), np.pi)

@njit_if_available(parallel=True, cache=True)
def cvtStokesToIntensity(img_stokes):
    """
    Convert stokes vector image to intensity image

    Parameters
    ----------
    img_stokes : np.ndarray, (height, width, 3)
        Stokes vector image

    Returns
    -------
    img_intensity : np.ndarray, (height, width)
        Intensity image
    """
    S0 = img_stokes[..., 0]
    return S0*0.5

@njit_if_available(parallel=True, cache=True)
def cvtStokesToDiffuse(img_stokes):
    """
    Convert stokes vector image to diffuse image

    Parameters
    ----------
    img_stokes : np.ndarray, (height, width, 3)
        Stokes vector image

    Returns
    -------
    img_diffuse : np.ndarray, (height, width)
        Diffuse image
    """
    Imin = cvtStokesToImin(img_stokes)
    return 1.0*Imin

@njit_if_available(parallel=True, cache=True)
def cvtStokesToSpecular(img_stokes):
    """
    Convert stokes vector image to specular image

    Parameters
    ----------
    img_stokes : np.ndarray, (height, width, 3)
        Stokes vector image

    Returns
    -------
    img_specular : np.ndarray, (height, width)
        Specular image
    """
    S1 = img_stokes[..., 1]
    S2 = img_stokes[..., 2]
    return np.sqrt(S1**2+S2**2) #same as Imax-Imin

@njit_if_available(parallel=True, cache=True)
def cvtStokesToDoP(img_stokes):
    """
    Convert stokes vector image to DoP (Degree of Polarization) image

    Parameters
    ----------
    img_stokes : np.ndarray, (height, width, 3)
        Stokes vector image

    Returns
    -------
    img_DoP : np.ndarray, (height, width)
        DoP image ∈ [0, 1]
    """
    S0 = img_stokes[..., 0]
    S1 = img_stokes[..., 1]
    S2 = img_stokes[..., 2]
    S3 = img_stokes[..., 3]
    return np.sqrt(S1**2+S2**2+S3**2)/S0

@njit_if_available(parallel=True, cache=True)
def cvtStokesToEllipticityAngle(img_stokes):
    """
    Convert stokes vector image to ellipticity angle image

    Parameters
    ----------
    img_stokes : np.ndarray, (height, width, 3)
        Stokes vector image

    Returns
    -------
    img_EllipticityAngle : np.ndarray, (height, width)
        ellipticity angle image ∈ [-pi/4, pi/4]
    """
    S1 = img_stokes[..., 1]
    S2 = img_stokes[..., 2]
    S3 = img_stokes[..., 3]
    return 0.5*np.arctan2(S3, np.sqrt(S1**2+S2**2))

@njit_if_available(parallel=True, cache=True)
def cvtStokesToDoCP(img_stokes):
    """
    Convert stokes vector image to DoCP (Degree of Circular Polarization) image

    Parameters
    ----------
    img_stokes : np.ndarray, (height, width, 3)
        Stokes vector image

    Returns
    -------
    img_DoCP : np.ndarray, (height, width)
        DoCP image ∈ [-1, 1]
    """
    S0 = img_stokes[..., 0]
    S3 = img_stokes[..., 3]
    return S3 / S0

def applyColorToAoLP(img_AoLP, saturation=1.0, value=1.0):
    """
    Apply color map to AoLP image
    The color map is based on HSV

    Parameters
    ----------
    img_AoLP : np.ndarray, (height, width)
        AoLP image. The range is from 0.0 to pi.
    
    saturation : float or np.ndarray, (height, width)
        Saturation part (optional).
        If you pass DoLP image (img_DoLP) as an argument, you can modulate it by DoLP.

    value : float or np.ndarray, (height, width)
        Value parr (optional).
        If you pass DoLP image (img_DoLP) as an argument, you can modulate it by DoLP.
    """
    img_ones = np.ones_like(img_AoLP)

    img_hue = (np.mod(img_AoLP, np.pi)/np.pi*179).astype(np.uint8) # 0~pi -> 0~179
    img_saturation = np.clip(img_ones*saturation*255, 0, 255).astype(np.uint8)
    img_value = np.clip(img_ones*value*255, 0, 255).astype(np.uint8)
    
    img_hsv = cv2.merge([img_hue, img_saturation, img_value])
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return img_bgr
