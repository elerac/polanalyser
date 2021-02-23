import cv2
import numpy as np

from enum import IntEnum, auto
class ColorConversionCodes(IntEnum):
    COLOR_IMX250MZR2MonoPola = auto()
    COLOR_IMX250MYR2ColorPola = auto()

def __demosaicing_float(img_raw):
    """
    Polarization demosaicing for arbitrary type
   
    cv2.cvtColor supports either uint8 or uint16 type. Float type bayer is demosaiced by this function.
    pros: slow
    cons: float available
    """
    height, width = img_raw.shape[:2]
    dtype = img_raw.dtype
    
    img_subsampled = np.zeros((height, width, 4), dtype=dtype)
    img_subsampled[0::2, 0::2, 0] = img_raw[0::2, 0::2]
    img_subsampled[0::2, 1::2, 1] = img_raw[0::2, 1::2]
    img_subsampled[1::2, 0::2, 2] = img_raw[1::2, 0::2]
    img_subsampled[1::2, 1::2, 3] = img_raw[1::2, 1::2]
    
    kernel = np.array([[1/4, 1/2, 1/4],
                       [1/2, 1.0, 1/2], 
                       [1/4, 1/2, 1/4]], dtype=np.float64)
    img_polarization = cv2.filter2D(img_subsampled, -1, kernel)
    return img_polarization[..., [3, 1, 0, 2]]

def __demosaicing_uint(img_raw):
    """
    Polarization demosaicing for np.uint8 or np.uint16 type
    """
    img_debayer_bg = cv2.cvtColor(img_raw, cv2.COLOR_BayerBG2BGR)
    img_debayer_gr = cv2.cvtColor(img_raw, cv2.COLOR_BayerGR2BGR)
    img_0,  _, img_90  = np.moveaxis(img_debayer_bg, -1, 0)
    img_45, _, img_135 = np.moveaxis(img_debayer_gr, -1, 0)
    img_polarization = np.array([img_0, img_45, img_90, img_135], dtype=img_raw.dtype)
    img_polarization = np.moveaxis(img_polarization, 0, -1)
    return img_polarization

def demosaicing(img_raw):
    """
    Polarization demosaicing

    Parameters
    ----------
    img_raw : np.ndarry, (height, width)
        RAW polarization image taken with polarizatin camera (e.g. IMX250MZR sensor)
    
    Returns
    -------
    img_polarization: np.ndarray, (height, width, 4)
        Dmosaiced image. 0-45-90-135.
    """
    if img_raw.dtype==np.uint8 or img_raw.dtype==np.uint16:
        return __demosaicing_uint(img_raw)
    else:
        return __demosaicing_float(img_raw)
