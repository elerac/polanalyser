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
    """Polarization demosaicing.

    This function separates a raw micro-polarizer image into four polarization
    images ordered as 0, 45, 90, and 135 degrees. The raw polarization pattern
    is assumed to be::

        90   45
        135   0

    For color-polarization sensors, the returned images use OpenCV's BGR
    channel order.

    Parameters
    ----------
    img_raw : np.ndarray
        Polarization image taken with polarization sensor (e.g. IMX250MZR, IMX250MYR sensor) with shape `(H, W)`.
    code : str, optional
        Color space conversion code.
        Supported values are `pa.COLOR_PolarRGB`, `pa.COLOR_PolarMono`,
        `pa.COLOR_PolarRGB_IGRI1`, `pa.COLOR_PolarRGB_IGRI2`, `pa.COLOR_PolarMono_IGRI1`,
        and `pa.COLOR_PolarMono_IGRI2`. The default is `pa.COLOR_PolarMono`.

    Returns
    -------
    demosaiced : list[np.ndarray] | np.ndarray
        List of demosaiced images. The shape of each image is `(H, W)` or `(H, W, 3)`.

    Notes
    -----
    The ``code`` argument selects both the sensor layout and the demosaicing
    method.

    ``pa.COLOR_PolarMono`` and ``pa.COLOR_PolarRGB``
        Bilinear interpolation demosaicing.
    ``pa.COLOR_PolarMono_IGRI1`` and ``pa.COLOR_PolarRGB_IGRI1``
        IGRI1 demosaicing [1]_.
    ``pa.COLOR_PolarMono_IGRI2`` and ``pa.COLOR_PolarRGB_IGRI2``
        IGRI2 demosaicing [2]_.

    Warnings
    --------
    Polanalyser's IGRI implementations are scale-invariant versions
    of the authors' official code, so the output is less dependent
    on the numeric range of the input image. As a side-effect, the numerical output
    can differ slightly from the original implementations.
    Use the authors' official implementation
    when exact benchmark reproduction is required.

    References
    ----------
    .. [1] M. Morimatsu, Y. Monno, M. Tanaka, and M. Okutomi,
       "Monochrome and Color Polarization Demosaicking Using Edge-Aware
       Residual Interpolation," 2020 IEEE International Conference on Image
       Processing (ICIP), pp. 2571-2575, 2020.

    .. [2] M. Morimatsu, Y. Monno, M. Tanaka, and M. Okutomi,
       "Monochrome and Color Polarization Demosaicking Based on
       Intensity-Guided Residual Interpolation," IEEE Sensors Journal,
       vol. 21, no. 23, pp. 26985-26996, 2021.
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

    code_list = ", ".join(_SUPPORTED_CODES)
    raise ValueError(f"Unsupported 'code': '{code}'. Supported codes are {code_list}.")
