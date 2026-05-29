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

    if img_raw.ndim != 2:
        raise ValueError(f"The dimension of the input image must be 2, not {img_raw.ndim} {img_raw.shape}")

    if not np.issubdtype(dtype, np.floating) and dtype not in [np.uint8, np.uint16]:
        raise TypeError(f"The dtype of input image must be `np.uint8`, `np.uint16`, or floating, not `{dtype}`")

    return demosaicing_func(img_raw)


def _demosaicing_mono(img_mpfa: np.ndarray) -> list[np.ndarray]:
    """Polarization demosaicing"""
    if np.issubdtype(img_mpfa.dtype, np.floating):
        return _demosaicing_mono_float(img_mpfa)

    img_debayer_bg = cv2.cvtColor(img_mpfa, cv2.COLOR_BayerBG2BGR)
    img_debayer_gr = cv2.cvtColor(img_mpfa, cv2.COLOR_BayerGR2BGR)
    img_000, _, img_090 = cv2.split(img_debayer_bg)
    img_045, _, img_135 = cv2.split(img_debayer_gr)
    return [img_000, img_045, img_090, img_135]


def _demosaicing_mono_float(img_mpfa: np.ndarray) -> list[np.ndarray]:
    """Naive polarization demosaicing for floating-point type."""
    img_debayer_bg = _demosaic_bayer_float(img_mpfa, "rggb")
    img_debayer_gr = _demosaic_bayer_float(img_mpfa, "gbrg")
    img_000, _, img_090 = cv2.split(img_debayer_bg)
    img_045, _, img_135 = cv2.split(img_debayer_gr)
    return [img_000, img_045, img_090, img_135]


def _demosaicing_color(img_cpfa: np.ndarray) -> list[np.ndarray]:
    """Color-polarization demosaicing"""
    if np.issubdtype(img_cpfa.dtype, np.floating):
        return _demosaicing_color_float(img_cpfa)

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


def _demosaicing_color_float(img_cpfa: np.ndarray) -> list[np.ndarray]:
    """Naive color-polarization demosaicing for floating-point type."""
    height, width = img_cpfa.shape[:2]
    work_dtype = _opencv_float_dtype(img_cpfa.dtype)

    img_mpfa_bgr = np.empty((height, width, 3), dtype=work_dtype)
    for j in range(2):
        for i in range(2):
            img_bayer_ij = img_cpfa[j::2, i::2].astype(work_dtype, copy=False)
            img_bgr_ij = _demosaic_bayer_float(img_bayer_ij, "rggb")
            img_mpfa_bgr[j::2, i::2] = img_bgr_ij

    img_bgr_000 = np.empty((height, width, 3), dtype=work_dtype)
    img_bgr_045 = np.empty((height, width, 3), dtype=work_dtype)
    img_bgr_090 = np.empty((height, width, 3), dtype=work_dtype)
    img_bgr_135 = np.empty((height, width, 3), dtype=work_dtype)
    for channel_index, img_mpfa in enumerate(cv2.split(img_mpfa_bgr)):
        img_000, img_045, img_090, img_135 = _demosaicing_mono_float(img_mpfa)
        img_bgr_000[..., channel_index] = img_000
        img_bgr_045[..., channel_index] = img_045
        img_bgr_090[..., channel_index] = img_090
        img_bgr_135[..., channel_index] = img_135

    return [
        img_bgr_000.astype(img_cpfa.dtype, copy=False),
        img_bgr_045.astype(img_cpfa.dtype, copy=False),
        img_bgr_090.astype(img_cpfa.dtype, copy=False),
        img_bgr_135.astype(img_cpfa.dtype, copy=False),
    ]


def _demosaic_bayer_float(img_bayer: np.ndarray, pattern: str) -> np.ndarray:
    """Naive Bayer demosaicing that mirrors OpenCV's bilinear Bayer phases."""
    height, width = img_bayer.shape[:2]
    work_dtype = _opencv_float_dtype(img_bayer.dtype)
    img_bayer_work = img_bayer.astype(work_dtype, copy=False)

    img_bgr = np.zeros((height, width, 3), dtype=work_dtype)
    if height < 3 or width < 3:
        return img_bgr.astype(img_bayer.dtype, copy=False)

    kernel_rb = np.array(
        [
            [0.25, 0.5, 0.25],
            [0.5, 1.0, 0.5],
            [0.25, 0.5, 0.25],
        ],
        dtype=work_dtype,
    )
    kernel_g = np.array(
        [
            [0.0, 0.25, 0.0],
            [0.25, 1.0, 0.25],
            [0.0, 0.25, 0.0],
        ],
        dtype=work_dtype,
    )

    masks = [np.zeros((height, width), dtype=work_dtype) for _ in range(3)]
    for index, color in enumerate(pattern):
        y = index // 2
        x = index % 2
        masks[{"b": 0, "g": 1, "r": 2}[color]][y::2, x::2] = 1.0

    for channel_index, mask in enumerate(masks):
        kernel = kernel_g if channel_index == 1 else kernel_rb
        img_masked = np.where(mask != 0.0, img_bayer_work, 0.0)
        img_bgr[..., channel_index] = cv2.filter2D(
            img_masked,
            -1,
            kernel,
            borderType=cv2.BORDER_CONSTANT,
        )

    img_bgr[0, :, :] = img_bgr[1, :, :]
    img_bgr[-1, :, :] = img_bgr[-2, :, :]
    img_bgr[:, 0, :] = img_bgr[:, 1, :]
    img_bgr[:, -1, :] = img_bgr[:, -2, :]
    return img_bgr.astype(img_bayer.dtype, copy=False)


def _opencv_float_dtype(dtype: np.dtype) -> np.dtype:
    dtype = np.dtype(dtype)
    if dtype == np.float64 or dtype.itemsize > np.dtype(np.float32).itemsize:
        return np.dtype(np.float64)
    return np.dtype(np.float32)
