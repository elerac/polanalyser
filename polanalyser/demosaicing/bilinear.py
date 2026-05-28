from collections.abc import Callable

import cv2
import numpy as np


def demosaicing_mono(img_mpfa: np.ndarray) -> list[np.ndarray] | np.ndarray:
    return _demosaicing(img_mpfa, _demosaicing_mono)


def demosaicing_color(img_cpfa: np.ndarray) -> list[np.ndarray] | np.ndarray:
    return _demosaicing(img_cpfa, _demosaicing_color)


def _demosaicing(
    img_raw: np.ndarray,
    demosaicing_func: Callable[[np.ndarray], list[np.ndarray]],
) -> list[np.ndarray] | np.ndarray:
    dtype = img_raw.dtype

    if np.issubdtype(dtype, np.floating):
        # Floating-point input is converted into uint16 for OpenCV demosaicing.
        # This preserves the existing stacked-ndarray return behavior for floats.
        scale = 65535.0 / np.nanmax(img_raw)
        img_raw_u16 = np.clip(img_raw * scale, 0, 65535).astype(np.uint16)
        img_demosaiced_u16 = _demosaicing(img_raw_u16, demosaicing_func)
        return [(img / scale).astype(img_raw.dtype) for img in img_demosaiced_u16]

    if dtype not in [np.uint8, np.uint16]:
        raise TypeError(f"The dtype of input image must be `np.uint8` or `np.uint16`, not `{dtype}`")

    if img_raw.ndim != 2:
        raise ValueError(f"The dimension of the input image must be 2, not {img_raw.ndim} {img_raw.shape}")

    return demosaicing_func(img_raw)


def _demosaicing_mono(img_mpfa: np.ndarray) -> list[np.ndarray]:
    """Polarization demosaicing for np.uint8 or np.uint16 type"""
    img_debayer_bg = cv2.cvtColor(img_mpfa, cv2.COLOR_BayerBG2BGR)
    img_debayer_gr = cv2.cvtColor(img_mpfa, cv2.COLOR_BayerGR2BGR)
    img_000, _, img_090 = cv2.split(img_debayer_bg)
    img_045, _, img_135 = cv2.split(img_debayer_gr)
    return [img_000, img_045, img_090, img_135]


def _demosaicing_color(img_cpfa: np.ndarray) -> list[np.ndarray]:
    """Color-polarization demosaicing for np.uint8 or np.uint16 type"""
    height, width = img_cpfa.shape[:2]

    # 1. Color demosaicing process
    img_mpfa_bgr = np.empty((height, width, 3), dtype=img_cpfa.dtype)
    for j in range(2):
        for i in range(2):
            # (i, j)
            # (0, 0) is 90,  (0, 1) is 45
            # (1, 0) is 135, (1, 1) is 0

            # Down sampling x2
            img_bayer_ij = img_cpfa[j::2, i::2]
            # Color demosaicing
            img_bgr_ij = cv2.cvtColor(img_bayer_ij, cv2.COLOR_BayerBG2BGR)
            # Up sampling x2
            img_mpfa_bgr[j::2, i::2] = img_bgr_ij

    # 2. Polarization demosaicing process
    img_bgr_000 = np.empty((height, width, 3), dtype=img_mpfa_bgr.dtype)
    img_bgr_045 = np.empty((height, width, 3), dtype=img_mpfa_bgr.dtype)
    img_bgr_090 = np.empty((height, width, 3), dtype=img_mpfa_bgr.dtype)
    img_bgr_135 = np.empty((height, width, 3), dtype=img_mpfa_bgr.dtype)
    for channel_index, img_mpfa in enumerate(cv2.split(img_mpfa_bgr)):
        img_000, img_045, img_090, img_135 = _demosaicing_mono(img_mpfa)
        img_bgr_000[..., channel_index] = img_000
        img_bgr_045[..., channel_index] = img_045
        img_bgr_090[..., channel_index] = img_090
        img_bgr_135[..., channel_index] = img_135

    return [img_bgr_000, img_bgr_045, img_bgr_090, img_bgr_135]
