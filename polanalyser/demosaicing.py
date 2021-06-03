import cv2
import numpy as np

COLOR_PolarRGB = "COLOR_PolarRGB"
COLOR_PolarMono = "COLOR_PolarMono"

def demosaicing(img_raw, code=COLOR_PolarMono):
    """Polarization demosaicing
    
    Parameters
    ----------
    img_raw : np.ndarry, (height, width)
      RAW polarization image taken with polarizatin camera 
      (e.g. IMX250MZR or IMX250MYR sensor)
    code : str (optional)
      COLOR_PolarMono or COLOR_PolarRGB
    
    Returns
    -------
    img_polarization: np.ndarray
        Dmosaiced image. 0-45-90-135.
    """
    if code == COLOR_PolarMono:
        if img_raw.dtype == np.uint8 or img_raw.dtype == np.uint16:
            return __demosaicing_mono_uint(img_raw)
        else:
            return __demosaicing_mono_float(img_raw)
    elif code == COLOR_PolarRGB:
        if img_raw.dtype == np.uint8 or img_raw.dtype == np.uint16:
            return __demosaicing_color(img_raw)
        else:
            raise TypeError("dtype of `img_raw` must be np.uint8 or np.uint16")
    else:
        raise ValueError(f"`code` must be {COLOR_PolarMono} or {COLOR_PolarRGB}")

def __demosaicing_mono_uint(img_mpfa):
    """Polarization demosaicing for np.uint8 or np.uint16 type
    """
    img_debayer_bg = cv2.cvtColor(img_mpfa, cv2.COLOR_BayerBG2BGR)
    img_debayer_gr = cv2.cvtColor(img_mpfa, cv2.COLOR_BayerGR2BGR)
    img_0,  _, img_90  = np.moveaxis(img_debayer_bg, -1, 0)
    img_45, _, img_135 = np.moveaxis(img_debayer_gr, -1, 0)
    img_polarization = np.array([img_0, img_45, img_90, img_135], dtype=img_mpfa.dtype)
    img_polarization = np.moveaxis(img_polarization, 0, -1)
    return img_polarization

def __demosaicing_mono_float(img_mpfa):
    """Polarization demosaicing for arbitrary type
   
    cv2.cvtColor supports either uint8 or uint16 type. 
    Float type bayer is demosaiced by this function.
    
    Notes
    -----
    pros: slow
    cons: float available
    """
    height, width = img_mpfa.shape[:2]
    img_subsampled = np.zeros((height, width, 4), dtype=img_mpfa.dtype)
    
    img_subsampled[0::2, 0::2, 0] = img_mpfa[0::2, 0::2]
    img_subsampled[0::2, 1::2, 1] = img_mpfa[0::2, 1::2]
    img_subsampled[1::2, 0::2, 2] = img_mpfa[1::2, 0::2]
    img_subsampled[1::2, 1::2, 3] = img_mpfa[1::2, 1::2]
    
    kernel = np.array([[1/4, 1/2, 1/4],
                       [1/2, 1.0, 1/2], 
                       [1/4, 1/2, 1/4]])
    
    img_polarization = cv2.filter2D(img_subsampled, -1, kernel)
    
    return img_polarization[..., [3, 1, 0, 2]]

def __demosaicing_color(img_cpfa):
    """Color-Polarization demosaicing for np.uint8 or np.uint16 type
    """
    height, width = img_cpfa.shape[:2]

    # 1. Color demosaicing process
    img_mpfa_bgr = np.empty((height, width, 3), dtype=img_cpfa.dtype)
    for j in range(2):
        for i in range(2):
            # (i, j)
            # (0, 0) is 90,  (0, 1) is 45
            # (1, 0) is 135, (1, 1) is 0

            # Down sampling ↓2
            img_bayer_ij = img_cpfa[j::2, i::2]
            # Color demosaicking
            img_bgr_ij = cv2.cvtColor(img_bayer_ij, cv2.COLOR_BayerBG2BGR)
            # Up samping ↑2
            img_mpfa_bgr[j::2, i::2] = img_bgr_ij

    # 2. Polarization demosaicing process
    img_bgr_polarization = np.empty((height, width, 3, 4), dtype=img_mpfa_bgr.dtype)
    for i, img_mpfa in enumerate(cv2.split(img_mpfa_bgr)):
        img_demosaiced = demosaicing(img_mpfa, COLOR_PolarMono)
        img_bgr_polarization[..., i, :] = img_demosaiced

    return img_bgr_polarization
