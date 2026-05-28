import numpy as np

from .bilinear import demosaicing_color, demosaicing_mono
from .igri import (
    demosaicing_color_igri1,
    demosaicing_color_igri2,
    demosaicing_mono_igri1,
    demosaicing_mono_igri2,
)

# Bilinear interpolation
COLOR_PolarRGB = "COLOR_PolarRGB"
COLOR_PolarMono = "COLOR_PolarMono"

# IGRI interpolation
COLOR_PolarRGB_IGRI1 = "COLOR_PolarRGB_IGRI1"
COLOR_PolarRGB_IGRI2 = "COLOR_PolarRGB_IGRI2"
COLOR_PolarMono_IGRI1 = "COLOR_PolarMono_IGRI1"
COLOR_PolarMono_IGRI2 = "COLOR_PolarMono_IGRI2"

_SUPPORTED_CODES = [
    COLOR_PolarRGB,
    COLOR_PolarMono,
    COLOR_PolarRGB_IGRI1,
    COLOR_PolarRGB_IGRI2,
    COLOR_PolarMono_IGRI1,
    COLOR_PolarMono_IGRI2,
]


def demosaicing(img_raw: np.ndarray, code: str = COLOR_PolarMono) -> list[np.ndarray] | np.ndarray:
    """Polarization demosaicing

    Parameters
    ----------
    img_raw : np.ndarray
        Polarization image taken with polarization sensor (e.g. IMX250MZR, IMX250MYR sensor). The shape is (height, width).
        Supported dtypes are defined by each demosaicing method.
    code : str, optional
        Color space conversion code. Supported values are `pa.COLOR_PolarRGB`, `pa.COLOR_PolarMono`,
        `pa.COLOR_PolarRGB_IGRI1`, `pa.COLOR_PolarRGB_IGRI2`, `pa.COLOR_PolarMono_IGRI1`,
        and `pa.COLOR_PolarMono_IGRI2`. The default is `pa.COLOR_PolarMono`.

    Returns
    -------
    img_demosaiced_list : list[np.ndarray] | np.ndarray
        List of demosaiced images. The shape of each image is (height, width) or (height, width, 3).
    """
    if code == COLOR_PolarRGB:
        return demosaicing_color(img_raw)

    if code == COLOR_PolarMono:
        return demosaicing_mono(img_raw)

    if code == COLOR_PolarRGB_IGRI1:
        return demosaicing_color_igri1(img_raw)

    if code == COLOR_PolarRGB_IGRI2:
        return demosaicing_color_igri2(img_raw)

    if code == COLOR_PolarMono_IGRI1:
        return demosaicing_mono_igri1(img_raw)

    if code == COLOR_PolarMono_IGRI2:
        return demosaicing_mono_igri2(img_raw)

    raise ValueError(f"Unsupported 'code': '{code}'. Supported codes are {", ".join(_SUPPORTED_CODES)}")
