import cv2
import numpy as np
from numba import njit

@njit(parallel=True, cache=True)
def __calcStokesPolaCam(images):
    """
    Calculate stokes vector from captured images and linear polarizer angles

    This function is the same as __calcStokesArbitrary() when radians is [0, np.pi/4, np.pi/2, np.pi*3/4]. A_pinv is very simple and stokes calculation can be written in a simpler form.
    As a result, it is about x4 faster than __calcStokesArbitrary(), thanks to the JIT compilation and parallelization of Numba.
    """
    height, width = images.shape[:2]
    img_stokes = np.empty((height, width, 3))
    img_stokes[:,:,0] = 0.5*images[:,:,0] + 0.5*images[:,:,1] + 0.5*images[:,:,2] + 0.5*images[:,:,3]
    img_stokes[:,:,1] = 1.0*images[:,:,0] - 1.0*images[:,:,2]
    img_stokes[:,:,2] = 1.0*images[:,:,1] - 1.0*images[:,:,3]
    return img_stokes

def __calcStokesArbitrary(images, radians):
    """
    Calculate stokes vector from captured images and linear polarizer angles
    """
    A = 0.5*np.array([np.ones_like(radians), np.cos(2*radians), np.sin(2*radians)]).T #(depth, 3)
    A_pinv = np.linalg.inv(A.T @ A) @ A.T #(3, depth)
    img_stokes = np.tensordot(A_pinv, images, axes=(1,2)).transpose(1, 2, 0) #(height, width, 3)
    return img_stokes

def calcStokes(images, radians):
    """
    Calculate stokes vector from captured images and linear polarizer angles

    Parameters
    ----------
    images : np.ndarray, (height, width, N)
        Captured images
    radians : np.ndarray, (N,)
        polarizer angles

    Returns
    -------
    img_stokes : np.ndarray, (height, width, 3)
        Calculated stokes vector image
    """
    if np.all(radians==np.array([0, np.pi/4, np.pi/2, np.pi*3/4])):
        # Special case when radians is [0, np.pi/4, np.pi/2, np.pi*3/4]
        return __calcStokesPolaCam(images)
    else:
        # Common case arbitrary radians
        return __calcStokesArbitrary(images, radians)


@njit(parallel=True, cache=True)
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
    S0 = img_stokes[:,:,0]
    S1 = img_stokes[:,:,1]
    S2 = img_stokes[:,:,2]
    return (S0+np.sqrt(S1**2+S2**2))*0.5

@njit(parallel=True, cache=True)
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
    S0 = img_stokes[:,:,0]
    S1 = img_stokes[:,:,1]
    S2 = img_stokes[:,:,2]
    return (S0-np.sqrt(S1**2+S2**2))*0.5

@njit(parallel=True, cache=True)
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
        DoLP image
    """
    S0 = img_stokes[:,:,0]
    S1 = img_stokes[:,:,1]
    S2 = img_stokes[:,:,2]
    return np.sqrt(S1**2+S2**2)/S0

@njit(parallel=True, cache=True)
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
        AoLP image
    """
    S1 = img_stokes[:,:,1]
    S2 = img_stokes[:,:,2]
    return np.mod(0.5*np.arctan2(S2, S1), np.pi)

@njit(parallel=True, cache=True)
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
    S0 = img_stokes[:,:,0]
    return S0*0.5

@njit(parallel=True, cache=True)
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

@njit(parallel=True, cache=True)
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
    S1 = img_stokes[:,:,1]
    S2 = img_stokes[:,:,2]
    return np.sqrt(S1**2+S2**2) #same as Imax-Imin

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
