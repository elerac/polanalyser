import cv2
import numpy as np
from numba import njit, float64

def calcStokes(images, radians):
    A = np.array([np.ones_like(radians), np.cos(2*radians), np.sin(2*radians)]).T #(depth, 3)
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


def calcMuller(images, radians_light, radians_camera):
    cos_light  = np.cos(2*radians_light)
    sin_light  = np.sin(2*radians_light)
    cos_camera = np.cos(2*radians_camera)
    sin_camera = np.sin(2*radians_camera)
    A = np.array([np.ones_like(radians_light), cos_light, sin_light, cos_camera, cos_camera*cos_light, cos_camera*sin_light, sin_camera, sin_camera*cos_light, sin_camera*sin_light]).T
    A_pinv = np.linalg.inv(A.T @ A) @ A.T #(9, depth)
    img_muller = np.tensordot(A_pinv, images, axes=(1,2)).transpose(1, 2, 0) #(height, width, 9)
    #height, width, _ = images.shape
    #img_muller_2D = np.reshape(img_muller, (height, width, 3, 3))
    return img_muller
