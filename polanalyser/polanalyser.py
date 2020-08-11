import cv2
import numpy as np
from numba import njit, float64, uint8, prange

@njit(parallel=True)
def calcStokesPolaCam(images):
    """
    This function is the same as calcStokes() when radians is [0, np.pi/4, np.pi/2, np.pi*3/4].
    If radians is [0, np.pi/4, np.pi/2, np.pi*3/4], then A_pinv is very simple and stokes calculation can be written in a simpler form.
    As a result, it is about x4 faster than calcStokes(), thanks to the JIT compilation and parallelization of Numba.
    """
    height, width = images.shape[:2]
    img_stokes = np.empty((height, width, 3))
    img_stokes[:,:,0] = 0.5*images[:,:,0] + 0.5*images[:,:,1] + 0.5*images[:,:,2] + 0.5*images[:,:,3]
    img_stokes[:,:,1] = 1.0*images[:,:,0] - 1.0*images[:,:,2]
    img_stokes[:,:,2] = 1.0*images[:,:,1] - 1.0*images[:,:,3]
    return img_stokes

def calcStokes(images, radians):
    if np.all(radians==np.array([0, np.pi/4, np.pi/2, np.pi*3/4])): return calcStokesPolaCam(images)

    A = 0.5*np.array([np.ones_like(radians), np.cos(2*radians), np.sin(2*radians)]).T #(depth, 3)
    A_pinv = np.linalg.inv(A.T @ A) @ A.T #(3, depth)
    img_stokes = np.tensordot(A_pinv, images, axes=(1,2)).transpose(1, 2, 0) #(height, width, 3)
    return img_stokes

@njit(float64[:,:](float64[:,:,:]), parallel=True)
def cvtStokesToImax(img_stokes):
    S0 = img_stokes[:,:,0]
    S1 = img_stokes[:,:,1]
    S2 = img_stokes[:,:,2]
    return (S0+np.sqrt(S1**2+S2**2))*0.5

@njit(float64[:,:](float64[:,:,:]), parallel=True)
def cvtStokesToImin(img_stokes):
    S0 = img_stokes[:,:,0]
    S1 = img_stokes[:,:,1]
    S2 = img_stokes[:,:,2]
    return (S0-np.sqrt(S1**2+S2**2))*0.5

@njit(float64[:,:](float64[:,:,:]), parallel=True)
def cvtStokesToDoLP(img_stokes):
    S0 = img_stokes[:,:,0]
    S1 = img_stokes[:,:,1]
    S2 = img_stokes[:,:,2]
    return np.sqrt(S1**2+S2**2)/S0

@njit(float64[:,:](float64[:,:,:]), parallel=True)
def cvtStokesToAoLP(img_stokes):
    S1 = img_stokes[:,:,1]
    S2 = img_stokes[:,:,2]
    return np.mod(0.5*np.arctan2(S2, S1), np.pi)

@njit(float64[:,:](float64[:,:,:]), parallel=True)
def cvtStokesToDiffuse(img_stokes):
    Imin = cvtStokesToImin(img_stokes)
    return 2*Imin

@njit(float64[:,:](float64[:,:,:]), parallel=True)
def cvtStokesToSpecular(img_stokes):
    Imax = cvtStokesToImax(img_stokes)
    Imin = cvtStokesToImin(img_stokes)
    return Imax-Imin

def applyLightColorToAoLP(img_AoLP, img_DoLP=None):
    img_ones = (np.ones_like(img_AoLP) * 255).astype(np.uint8)
    img_normalized_AoLP = (np.mod(img_AoLP, np.pi)/np.pi*179).astype(np.uint8) # 0~pi -> 0~179
    if img_DoLP is not None:
        img_normalized_DoLP = np.clip(img_DoLP*255, 0, 255).astype(np.uint8)
    else:
        img_normalized_DoLP = img_ones
    
    img_hsv = cv2.merge([img_normalized_AoLP, img_normalized_DoLP, img_ones])
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return img_bgr

def applyDarkColorToAoLP(img_AoLP, img_DoLP=None):
    img_ones = (np.ones_like(img_AoLP) * 255).astype(np.uint8)
    img_normalized_AoLP = (np.mod(img_AoLP, np.pi)/np.pi*179).astype(np.uint8) # 0~pi -> 0~179
    if img_DoLP is not None:
        img_normalized_DoLP = np.clip(img_DoLP*255, 0, 255).astype(np.uint8)
    else:
        img_normalized_DoLP = img_ones
    
    img_hsv = cv2.merge([img_normalized_AoLP, img_ones, img_normalized_DoLP])
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return img_bgr


def calcMueller(images, radians_light, radians_camera):
    cos_light  = np.cos(2*radians_light)
    sin_light  = np.sin(2*radians_light)
    cos_camera = np.cos(2*radians_camera)
    sin_camera = np.sin(2*radians_camera)
    A = np.array([np.ones_like(radians_light), cos_light, sin_light, cos_camera, cos_camera*cos_light, cos_camera*sin_light, sin_camera, sin_camera*cos_light, sin_camera*sin_light]).T
    A_pinv = np.linalg.inv(A.T @ A) @ A.T #(9, depth)
    img_mueller = np.tensordot(A_pinv, images, axes=(1,2)).transpose(1, 2, 0) #(height, width, 9)
    #height, width, _ = images.shape
    #img_mueller_2D = np.reshape(img_mueller, (height, width, 3, 3))
    return img_mueller
