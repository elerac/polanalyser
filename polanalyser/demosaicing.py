import cv2
import numpy as np

def __demosaicing_float(bayer):
    """
    Polarization demosaicing for arbitrary type
    
    cv2.cvtColor supports either uint8 or uint16 type. Float type bayer is demosaiced by this function.
    pros: slow
    cons: float available
    """
    kernel = np.array([[1/4, 1/2, 1/4],[1/2, 1, 1/2],[1/4, 1/2, 1/4]], dtype=np.float64)
    
    height, width = bayer.shape
    pre_bayer = np.zeros((height, width, 4), dtype=bayer.dtype)
    pre_bayer[0::2, 0::2, 0] = bayer[0::2, 0::2]
    pre_bayer[0::2, 1::2, 1] = bayer[0::2, 1::2]
    pre_bayer[1::2, 0::2, 2] = bayer[1::2, 0::2]
    pre_bayer[1::2, 1::2, 3] = bayer[1::2, 1::2]
    
    img_polarization = cv2.filter2D(pre_bayer, -1, kernel)
    return img_polarization[:,:,[3, 1, 0, 2]]

def __demosaicing_uint(bayer):
    """
    Polarization demosaicing for np.uint8 or np.uint16 type
    """
    BG = cv2.cvtColor(bayer, cv2.COLOR_BayerBG2BGR)
    GR = cv2.cvtColor(bayer, cv2.COLOR_BayerGR2BGR)
    img_polarization = np.array((BG[:,:,0], GR[:,:,0], BG[:,:,2], GR[:,:,2])).transpose(1, 2, 0)
    return img_polarization

def demosaicing(img_bayer):
    """
    Polarization demosaicing

    Parameters
    ----------
    img_bayer : np.ndarry, (height, width)
        RAW polarization image taken with polarizatin camera (e.g. IMX250MZR sensor)
    
    Returns
    -------
    img_polarization: np.ndarray, (height, width, 4)
        Dmosaiced image. 0-45-90-135.
    """
    if img_bayer.dtype==np.uint8 or img_bayer.dtype==np.uint16:
        return __demosaicing_uint(img_bayer)
    else:
        return __demosaicing_float(img_bayer)
