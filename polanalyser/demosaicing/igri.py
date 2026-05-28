"""IGRI1/IGRI2 polarization demosaicing with scale-invariant modifications.

This implementation follows the IGRI1 [1]_ / IGRI2 [2]_ structure,
while modifying several numerical details of the reference implementation
to make the demosaicing procedure scale-invariant across a wider range of input data,
without relying on fixed absolute thresholds.

The main differences are:
- Inverse-cost weights use explicit zero-cost limit handling instead of
  additive absolute epsilons.
- RI slope estimation uses a zero-denominator pseudoinverse case instead of
  fixed ridge-like stabilization.
- Residual-cost weighting avoids fixed absolute floors.
- Intermediate RI and guide estimates are not clipped during reconstruction.
- IGRI2 diagonal guide construction tracks valid samples so padded canvas
  locations are not treated as observations.

These modifications may lead to numerical results that differ slightly from those of the authors' reference implementation.
Use the reference implementation when exact benchmark reproduction is required.

References
----------
.. [1] M. Morimatsu, Y. Monno, M. Tanaka, and M. Okutomi,
       "Monochrome and Color Polarization Demosaicking Using Edge-Aware
       Residual Interpolation," 2020 IEEE International Conference on Image
       Processing (ICIP), pp. 2571-2575, 2020.
       doi: 10.1109/ICIP40778.2020.9191085

.. [2] M. Morimatsu, Y. Monno, M. Tanaka, and M. Okutomi,
       "Monochrome and Color Polarization Demosaicking Based on
       Intensity-Guided Residual Interpolation," IEEE Sensors Journal,
       vol. 21, no. 23, pp. 26985-26996, 2021.
       doi: 10.1109/JSEN.2021.3121884
"""

import numpy as np

BAYER_PATTERNS = ("rggb", "grbg", "gbrg", "bggr")
DEFAULT_SIGMA = 1.0
ROUND_OFF_ZERO_REL = 1e-9


def demosaicing_mono_igri1(img_mpfa: np.ndarray) -> list[np.ndarray]:
    return _demosaicing_mono(img_mpfa, "igri1")


def demosaicing_mono_igri2(img_mpfa: np.ndarray) -> list[np.ndarray]:
    return _demosaicing_mono(img_mpfa, "igri2")


def demosaicing_color_igri1(img_cpfa: np.ndarray) -> list[np.ndarray]:
    return _demosaicing_color(img_cpfa, "igri1")


def demosaicing_color_igri2(img_cpfa: np.ndarray) -> list[np.ndarray]:
    return _demosaicing_color(img_cpfa, "igri2")


def imfilter(src: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """MATLAB imfilter(..., 'replicate') with correlation semantics."""
    from scipy import ndimage

    src = np.asarray(src, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim == 1:
        kernel = kernel.reshape(1, -1)
    if kernel.ndim != 2:
        raise ValueError("kernel must be 1D or 2D")
    if src.ndim == 3:
        return np.stack([imfilter(src[..., channel], kernel) for channel in range(src.shape[2])], axis=2)
    return ndimage.correlate(src, kernel, mode="nearest")


def boxfilter(src: np.ndarray, h: int, v: int) -> np.ndarray:
    """Windowed sum matching the MATLAB TIP_RI/boxfilter.m implementation."""
    src = np.asarray(src, dtype=np.float64)
    if src.ndim == 3:
        return np.stack([boxfilter(src[..., channel], h, v) for channel in range(src.shape[2])], axis=2)

    height, width = src.shape
    if h < 0 or v < 0:
        raise ValueError("boxfilter radii must be non-negative")
    if (v and height <= 2 * v) or (h and width <= 2 * h):
        raise ValueError(f"boxfilter radius ({h}, {v}) is too large for image shape {src.shape}")

    dst = np.zeros_like(src, dtype=np.float64)

    if v != 0:
        cumulative = np.cumsum(src, axis=0)
        dst[: v + 1, :] = cumulative[v : 2 * v + 1, :]
        dst[v + 1 : height - v, :] = cumulative[2 * v + 1 : height, :] - cumulative[: height - 2 * v - 1, :]
        dst[height - v : height, :] = cumulative[height - 1 : height, :] - cumulative[height - 2 * v - 1 : height - v - 1, :]

    if h != 0:
        cumulative = np.cumsum(dst if v != 0 else src, axis=1)
        dst[:, : h + 1] = cumulative[:, h : 2 * h + 1]
        dst[:, h + 1 : width - h] = cumulative[:, 2 * h + 1 : width] - cumulative[:, : width - 2 * h - 1]
        dst[:, width - h : width] = cumulative[:, width - 1 : width] - cumulative[:, width - 2 * h - 1 : width - h - 1]

    if h == 0 and v == 0:
        return src.copy()
    return dst


def gaussian_kernel(size: tuple[int, int], sigma: float) -> np.ndarray:
    rows, cols = size
    y = np.arange(rows, dtype=np.float64) - (rows - 1) / 2.0
    x = np.arange(cols, dtype=np.float64) - (cols - 1) / 2.0
    xx, yy = np.meshgrid(x, y)
    kernel = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    total = float(np.sum(kernel))
    if total > 0:
        kernel /= total
    return kernel


def bayer_mask(shape: tuple[int, int], pattern: str) -> np.ndarray:
    pattern = pattern.lower()
    if pattern not in BAYER_PATTERNS:
        raise ValueError(f"unsupported Bayer pattern: {pattern!r}")

    height, width = shape
    channel = {"r": 0, "g": 1, "b": 2}
    channels = [channel[name] for name in pattern]
    mask = np.zeros((height, width, 3), dtype=np.float64)
    mask[0::2, 0::2, channels[0]] = 1.0
    mask[0::2, 1::2, channels[1]] = 1.0
    mask[1::2, 0::2, channels[2]] = 1.0
    mask[1::2, 1::2, channels[3]] = 1.0
    return mask


def mask_gr_gb(shape: tuple[int, int], pattern: str) -> tuple[np.ndarray, np.ndarray]:
    pattern = pattern.lower()
    if pattern not in BAYER_PATTERNS:
        raise ValueError(f"unsupported Bayer pattern: {pattern!r}")

    height, width = shape
    mask_gr = np.zeros((height, width), dtype=np.float64)
    mask_gb = np.zeros((height, width), dtype=np.float64)
    if pattern == "grbg":
        mask_gr[0::2, 0::2] = 1.0
        mask_gb[1::2, 1::2] = 1.0
    elif pattern == "rggb":
        mask_gr[0::2, 1::2] = 1.0
        mask_gb[1::2, 0::2] = 1.0
    elif pattern == "gbrg":
        mask_gb[0::2, 0::2] = 1.0
        mask_gr[1::2, 1::2] = 1.0
    elif pattern == "bggr":
        mask_gb[0::2, 1::2] = 1.0
        mask_gr[1::2, 0::2] = 1.0
    return mask_gr, mask_gb


def create_polar_masks(shape: tuple[int, int]) -> tuple[np.ndarray, ...]:
    height, width = shape
    mask_90 = np.zeros((height, width), dtype=np.float64)
    mask_45 = np.zeros((height, width), dtype=np.float64)
    mask_135 = np.zeros((height, width), dtype=np.float64)
    mask_0 = np.zeros((height, width), dtype=np.float64)
    mask_90[0::2, 0::2] = 1.0
    mask_45[0::2, 1::2] = 1.0
    mask_135[1::2, 0::2] = 1.0
    mask_0[1::2, 1::2] = 1.0
    return mask_0, mask_45, mask_90, mask_135


def diagonal_maps(height: int, width: int) -> tuple[tuple[int, int], tuple[int, int], np.ndarray, ...]:
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError("IGRI2 requires even image dimensions")

    rows, cols = np.indices((height, width))
    even = ((rows + cols) % 2) == 0
    odd = ~even
    half_height = height // 2

    even_r = ((rows + cols) // 2)[even]
    even_c = ((cols - rows) // 2 + half_height - 1)[even]
    odd_r = ((rows + cols - 1) // 2)[odd]
    odd_c = ((cols - rows - 1) // 2 + half_height)[odd]

    even_shape = (max(height, int(even_r.max()) + 1), max(width, int(even_c.max()) + 1))
    odd_shape = (max(height, int(odd_r.max()) + 1), max(width, int(odd_c.max()) + 1))

    if np.any(even_r < 0) or np.any(even_c < 0) or np.any(odd_r < 0) or np.any(odd_c < 0) or np.any(even_r >= even_shape[0]) or np.any(even_c >= even_shape[1]) or np.any(odd_r >= odd_shape[0]) or np.any(odd_c >= odd_shape[1]):
        raise ValueError("IGRI2 diagonal coordinates fell outside the expanded canvas")

    return even_shape, odd_shape, even, odd, even_r, even_c, odd_r, odd_c


def _demosaicing_mono(img_mpfa: np.ndarray, method: str) -> list[np.ndarray]:
    raw_float = _as_float_image(img_mpfa)
    height, width = raw_float.shape
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(f"raw mono-polarization mosaic dimensions must be divisible by 2, got {width} x {height}.")

    img_mpfa_float = raw_float[..., np.newaxis]
    mask_0, mask_45, mask_90, mask_135 = create_polar_masks(raw_float.shape)
    if method == "igri1":
        img_list = _igri1(img_mpfa_float, mask_0, mask_45, mask_90, mask_135)
    elif method == "igri2":
        img_list = _igri2(img_mpfa_float, mask_0, mask_45, mask_90, mask_135)
    else:
        raise ValueError(f"unsupported method: {method!r}")
    return [img[..., 0] for img in img_list]


def _demosaicing_color(img_cpfa: np.ndarray, method: str) -> list[np.ndarray]:
    raw_float = _as_float_image(img_cpfa)
    height, width = raw_float.shape
    if height % 4 != 0 or width % 4 != 0:
        raise ValueError(f"raw RGB-polarization mosaic dimensions must be divisible by 4, got {width} x {height}.")

    img_mpfa_bgr = _color_polar_mosaic_to_bgr(raw_float, "rggb")
    mask_0, mask_45, mask_90, mask_135 = create_polar_masks(raw_float.shape)
    if method == "igri1":
        return list(_igri1(img_mpfa_bgr, mask_0, mask_45, mask_90, mask_135))
    if method == "igri2":
        return list(_igri2(img_mpfa_bgr, mask_0, mask_45, mask_90, mask_135))
    raise ValueError(f"unsupported method: {method!r}")


def _as_float_image(raw: np.ndarray) -> np.ndarray:
    if raw.ndim != 2:
        raise ValueError(f"The dimension of the input image must be 2, not {raw.ndim} {raw.shape}")
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.complexfloating):
        raise TypeError(f"The dtype of input image must be real numeric, not `{raw.dtype}`")
    return raw.astype(np.float64, copy=False)


def _divide_or_zero(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.zeros_like(np.asarray(numerator, dtype=np.float64), dtype=np.float64)
    np.divide(numerator, denominator, out=out, where=denominator != 0)
    return out


def _normalized_inverse_weights(costs: list[np.ndarray] | tuple[np.ndarray, ...], regularizer: float = 0.0) -> np.ndarray:
    """Return normalized inverse-cost weights using the epsilon->0 limit."""
    cost = np.stack([np.asarray(item, dtype=np.float64) for item in costs], axis=0)
    cost = np.maximum(cost, 0.0)
    if regularizer > 0:
        cost = cost + regularizer

    weights = np.zeros_like(cost, dtype=np.float64)
    finite_cost = np.where(np.isfinite(cost), cost, 0.0)
    cost_scale = np.max(finite_cost, axis=0)
    global_scale = float(np.max(finite_cost))
    cost_scale = np.maximum(cost_scale, global_scale)
    zero_tol = ROUND_OFF_ZERO_REL * cost_scale
    zero = cost <= zero_tol[np.newaxis, ...]
    zero_count = np.sum(zero, axis=0)

    np.divide(zero.astype(np.float64), zero_count[np.newaxis, ...], out=weights, where=zero_count[np.newaxis, ...] > 0)

    no_zero = zero_count == 0
    positive = np.isfinite(cost) & (~zero)
    inv = np.zeros_like(cost, dtype=np.float64)
    inv[positive] = 1.0 / cost[positive]
    inv_sum = np.sum(inv, axis=0)

    use_inverse = no_zero & (inv_sum > 0)
    np.divide(inv, inv_sum[np.newaxis, ...], out=weights, where=use_inverse[np.newaxis, ...])

    fallback = no_zero & (inv_sum == 0)
    if np.any(fallback):
        weights[:, fallback] = 1.0 / cost.shape[0]

    return weights


def _inverse_cost_window_average(
    value: np.ndarray,
    cost: np.ndarray,
    valid: np.ndarray,
    h: int,
    v: int,
    regularizer: float = 0.0,
) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)

    weighted_cost = np.maximum(cost, 0.0)
    if regularizer > 0:
        weighted_cost = np.where(valid, weighted_cost + regularizer, np.inf)

    finite_cost = np.where(valid & np.isfinite(weighted_cost), weighted_cost, 0.0)
    global_scale = float(np.max(finite_cost))
    local_scale = boxfilter(finite_cost, h, v)
    local_count = boxfilter((valid & np.isfinite(weighted_cost)).astype(np.float64), h, v)
    local_scale = _divide_or_zero(local_scale, local_count)
    zero_tol = ROUND_OFF_ZERO_REL * np.maximum.reduce((weighted_cost, local_scale, np.full_like(weighted_cost, global_scale)))
    zero = valid & (weighted_cost <= zero_tol)
    zero_count = boxfilter(zero.astype(np.float64), h, v)
    zero_num = boxfilter(value * zero, h, v)

    positive = valid & np.isfinite(weighted_cost) & (~zero)
    inv = np.zeros_like(weighted_cost, dtype=np.float64)
    inv[positive] = 1.0 / weighted_cost[positive]
    inv_sum = boxfilter(inv, h, v)
    inv_num = boxfilter(value * inv, h, v)

    out = np.zeros_like(value, dtype=np.float64)
    use_zero = zero_count > 0
    np.divide(zero_num, zero_count, out=out, where=use_zero)

    use_inverse = (~use_zero) & (inv_sum > 0)
    np.divide(inv_num, inv_sum, out=out, where=use_inverse)

    fallback_count = boxfilter(valid.astype(np.float64), h, v)
    fallback_num = boxfilter(value * valid, h, v)
    use_fallback = (~use_zero) & (~use_inverse) & (fallback_count > 0)
    np.divide(fallback_num, fallback_count, out=out, where=use_fallback)

    return out


def _normalized_filter(values: np.ndarray, valid: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    valid_f = np.asarray(valid, dtype=np.float64)
    num = imfilter(np.asarray(values, dtype=np.float64) * valid_f, kernel)
    den = imfilter(valid_f, kernel)
    return _divide_or_zero(num, den)


def _mosaic_from_cfa_masked(cfa: np.ndarray, pattern: str, valid_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfa = np.asarray(cfa, dtype=np.float64)
    mask = bayer_mask(cfa.shape, pattern)
    if valid_mask is None:
        valid = np.ones(cfa.shape, dtype=np.float64)
    else:
        valid = np.asarray(valid_mask, dtype=np.float64)
        if valid.shape != cfa.shape:
            raise ValueError(f"valid_mask shape {valid.shape} does not match CFA shape {cfa.shape}.")
    mask = mask * valid[..., np.newaxis]
    return cfa[..., np.newaxis] * mask, mask, valid


def _guidedfilter_mlri_weighted(
    guide: np.ndarray,
    reference: np.ndarray,
    reference_mask: np.ndarray,
    lap_guide: np.ndarray,
    lap_input: np.ndarray,
    input_mask: np.ndarray,
    h: int,
    v: int,
) -> np.ndarray:
    n_input = boxfilter(input_mask, h, v)
    numerator = boxfilter(lap_guide * lap_input * input_mask, h, v)
    denominator = boxfilter(lap_guide * lap_guide * input_mask, h, v)
    denominator_scale = float(np.max(np.where(np.isfinite(denominator), denominator, 0.0)))
    denominator = np.where(denominator <= ROUND_OFF_ZERO_REL * denominator_scale, 0.0, denominator)
    a = _divide_or_zero(numerator, denominator)

    n_ref = boxfilter(reference_mask, h, v)
    valid_ref = n_ref > 0
    sum_g = boxfilter(guide * reference_mask, h, v)
    sum_r = boxfilter(reference * reference_mask, h, v)
    mean_g = _divide_or_zero(sum_g, n_ref)
    mean_r = _divide_or_zero(sum_r, n_ref)
    b = np.where(valid_ref, mean_r - a * mean_g, 0.0)

    sum_g2 = boxfilter(guide * guide * reference_mask, h, v)
    sum_r2 = boxfilter(reference * reference * reference_mask, h, v)
    sum_gr = boxfilter(guide * reference * reference_mask, h, v)
    cost_sum = a * a * sum_g2 + 2.0 * a * b * sum_g + b * b * n_ref + sum_r2 - 2.0 * a * sum_gr - 2.0 * b * sum_r
    cost_scale = np.abs(a * a * sum_g2) + np.abs(2.0 * a * b * sum_g) + np.abs(b * b * n_ref) + np.abs(sum_r2) + np.abs(2.0 * a * sum_gr) + np.abs(2.0 * b * sum_r)
    near_zero = cost_sum <= ROUND_OFF_ZERO_REL * cost_scale
    cost_sum = np.where(near_zero, 0.0, cost_sum)
    cost = np.full_like(guide, np.inf, dtype=np.float64)
    cost[valid_ref] = np.maximum(cost_sum[valid_ref], 0.0) / n_ref[valid_ref]

    mean_a = _inverse_cost_window_average(a, cost, valid_ref, h, v)
    mean_b = _inverse_cost_window_average(b, cost, valid_ref, h, v)
    return mean_a * guide + mean_b


def _residual_interpolation(
    guide: np.ndarray,
    mosaic: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    h = 5
    v = 5
    laplacian = np.array(
        [
            [0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0],
            [-1, 0, 4, 0, -1],
            [0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0],
        ],
        dtype=np.float64,
    )
    lap_input = imfilter(mosaic, laplacian)
    lap_guide = imfilter(guide * mask, laplacian)
    tentative = _guidedfilter_mlri_weighted(guide, mosaic, mask, lap_guide, lap_input, mask, h, v)
    residual = mask * (mosaic - tentative)
    bilinear = np.array(
        [[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]],
        dtype=np.float64,
    )
    return imfilter(residual, bilinear) + tentative


def _green_interpolation(
    mosaic: np.ndarray,
    mask: np.ndarray,
    pattern: str,
    sigma: float,
    valid_pixels: np.ndarray | None = None,
) -> np.ndarray:
    sample_valid = np.sum(mask, axis=2) > 0
    if valid_pixels is None:
        valid_pixels = sample_valid
    else:
        valid_pixels = np.asarray(valid_pixels, dtype=bool)

    imask = (mask == 0).astype(np.float64)
    rawq = np.sum(mosaic, axis=2)
    mask_gr, mask_gb = mask_gr_gb(rawq.shape, pattern)
    mask_gr = mask_gr * sample_valid
    mask_gb = mask_gb * sample_valid

    kh_half = np.array([[0.5, 0.0, 0.5]], dtype=np.float64)
    kv_half = kh_half.T
    rawh = _normalized_filter(rawq, sample_valid, kh_half)
    rawv = _normalized_filter(rawq, sample_valid, kv_half)

    guide_gh = mosaic[..., 1] + rawh * mask[..., 0] + rawh * mask[..., 2]
    guide_rh = mosaic[..., 0] + rawh * mask_gr
    guide_bh = mosaic[..., 2] + rawh * mask_gb
    guide_gv = mosaic[..., 1] + rawv * mask[..., 0] + rawv * mask[..., 2]
    guide_rv = mosaic[..., 0] + rawv * mask_gb
    guide_bv = mosaic[..., 2] + rawv * mask_gr

    h = 3
    v = 3
    diff_kernel = np.array([[-1.0, 0.0, 2.0, 0.0, -1.0]], dtype=np.float64)

    dif_r = imfilter(mosaic[..., 0], diff_kernel)
    dif_gr = imfilter(guide_gh * mask[..., 0], diff_kernel)
    tentative_rh = _guidedfilter_mlri_weighted(guide_gh, mosaic[..., 0], mask[..., 0], dif_gr, dif_r, mask[..., 0], h, v)

    dif_gr = imfilter(mosaic[..., 1] * mask_gr, diff_kernel)
    dif_r = imfilter(guide_rh * mask_gr, diff_kernel)
    tentative_grh = _guidedfilter_mlri_weighted(guide_rh, mosaic[..., 1] * mask_gr, mask_gr, dif_r, dif_gr, mask_gr, h, v)

    dif_b = imfilter(mosaic[..., 2], diff_kernel)
    dif_gb = imfilter(guide_gh * mask[..., 2], diff_kernel)
    tentative_bh = _guidedfilter_mlri_weighted(guide_gh, mosaic[..., 2], mask[..., 2], dif_gb, dif_b, mask[..., 2], h, v)

    dif_gb = imfilter(mosaic[..., 1] * mask_gb, diff_kernel)
    dif_b = imfilter(guide_bh * mask_gb, diff_kernel)
    tentative_gbh = _guidedfilter_mlri_weighted(guide_bh, mosaic[..., 1] * mask_gb, mask_gb, dif_b, dif_gb, mask_gb, h, v)

    diff_kernel = diff_kernel.T
    dif_r = imfilter(mosaic[..., 0], diff_kernel)
    dif_gr = imfilter(guide_gv * mask[..., 0], diff_kernel)
    tentative_rv = _guidedfilter_mlri_weighted(guide_gv, mosaic[..., 0], mask[..., 0], dif_gr, dif_r, mask[..., 0], v, h)

    dif_gr = imfilter(mosaic[..., 1] * mask_gb, diff_kernel)
    dif_r = imfilter(guide_rv * mask_gb, diff_kernel)
    tentative_grv = _guidedfilter_mlri_weighted(guide_rv, mosaic[..., 1] * mask_gb, mask_gb, dif_r, dif_gr, mask_gb, v, h)

    dif_b = imfilter(mosaic[..., 2], diff_kernel)
    dif_gb = imfilter(guide_gv * mask[..., 2], diff_kernel)
    tentative_bv = _guidedfilter_mlri_weighted(guide_gv, mosaic[..., 2], mask[..., 2], dif_gb, dif_b, mask[..., 2], v, h)

    dif_gb = imfilter(mosaic[..., 1] * mask_gr, diff_kernel)
    dif_b = imfilter(guide_bv * mask_gr, diff_kernel)
    tentative_gbv = _guidedfilter_mlri_weighted(guide_bv, mosaic[..., 1] * mask_gr, mask_gr, dif_b, dif_gb, mask_gr, v, h)

    residual_grh = (mosaic[..., 1] - tentative_grh) * mask_gr
    residual_gbh = (mosaic[..., 1] - tentative_gbh) * mask_gb
    residual_rh = (mosaic[..., 0] - tentative_rh) * mask[..., 0]
    residual_bh = (mosaic[..., 2] - tentative_bh) * mask[..., 2]
    residual_grv = (mosaic[..., 1] - tentative_grv) * mask_gb
    residual_gbv = (mosaic[..., 1] - tentative_gbv) * mask_gr
    residual_rv = (mosaic[..., 0] - tentative_rv) * mask[..., 0]
    residual_bv = (mosaic[..., 2] - tentative_bv) * mask[..., 2]

    residual_grh = _normalized_filter(residual_grh, mask_gr, kh_half)
    residual_gbh = _normalized_filter(residual_gbh, mask_gb, kh_half)
    residual_rh = _normalized_filter(residual_rh, mask[..., 0], kh_half)
    residual_bh = _normalized_filter(residual_bh, mask[..., 2], kh_half)
    residual_grv = _normalized_filter(residual_grv, mask_gb, kv_half)
    residual_gbv = _normalized_filter(residual_gbv, mask_gr, kv_half)
    residual_rv = _normalized_filter(residual_rv, mask[..., 0], kv_half)
    residual_bv = _normalized_filter(residual_bv, mask[..., 2], kv_half)

    grh = (tentative_grh + residual_grh) * mask[..., 0]
    gbh = (tentative_gbh + residual_gbh) * mask[..., 2]
    rh = (tentative_rh + residual_rh) * mask_gr
    bh = (tentative_bh + residual_bh) * mask_gb
    grv = (tentative_grv + residual_grv) * mask[..., 0]
    gbv = (tentative_gbv + residual_gbv) * mask[..., 2]
    rv = (tentative_rv + residual_rv) * mask_gb
    bv = (tentative_bv + residual_bv) * mask_gr

    difh = mosaic[..., 1] + grh + gbh - mosaic[..., 0] - mosaic[..., 2] - rh - bh
    difv = mosaic[..., 1] + grv + gbv - mosaic[..., 0] - mosaic[..., 2] - rv - bv
    difh = np.where(valid_pixels, difh, 0.0)
    difv = np.where(valid_pixels, difv, 0.0)

    kh_grad = np.array([[1.0, 0.0, -1.0]], dtype=np.float64)
    kv_grad = kh_grad.T
    difh2 = np.abs(imfilter(difh, kh_grad))
    difv2 = np.abs(imfilter(difv, kv_grad))

    smooth3 = np.ones((3, 3), dtype=np.float64)
    wh = _normalized_filter(difh2, valid_pixels, smooth3)
    wv = _normalized_filter(difv2, valid_pixels, smooth3)

    kw = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    ke = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    kn = kw.T
    ks = ke.T
    cost_w = _normalized_filter(wh, valid_pixels, kw) ** 2
    cost_e = _normalized_filter(wh, valid_pixels, ke) ** 2
    cost_n = _normalized_filter(wv, valid_pixels, kn) ** 2
    cost_s = _normalized_filter(wv, valid_pixels, ks) ** 2

    hwin = gaussian_kernel((1, 9), sigma)
    ke_long = np.array([[0, 0, 0, 0, 1, 1, 1, 1, 1]], dtype=np.float64) * hwin
    kw_long = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0]], dtype=np.float64) * hwin
    ke_long /= np.sum(ke_long, axis=1, keepdims=True)
    kw_long /= np.sum(kw_long, axis=1, keepdims=True)
    ks_long = ke_long.T
    kn_long = kw_long.T

    difn = _normalized_filter(difv, valid_pixels, kn_long)
    difs = _normalized_filter(difv, valid_pixels, ks_long)
    difw = _normalized_filter(difh, valid_pixels, kw_long)
    dife = _normalized_filter(difh, valid_pixels, ke_long)

    weights = _normalized_inverse_weights((cost_n, cost_s, cost_w, cost_e))
    dif = weights[0] * difn + weights[1] * difs + weights[2] * difw + weights[3] * dife
    green = dif + rawq
    green = green * imask[..., 1] + rawq * mask[..., 1]
    return np.where(valid_pixels, green, 0.0)


def _residual_interpolation2(
    mosaic2d: np.ndarray,
    pattern: str,
    sigma: float,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    mosaic, mask, valid = _mosaic_from_cfa_masked(mosaic2d, pattern, valid_mask)
    return _green_interpolation(mosaic, mask, pattern, sigma, valid_pixels=valid.astype(bool))


def _demosaic_bayer_cfa(cfa: np.ndarray, pattern: str, sigma: float) -> np.ndarray:
    mosaic, mask, _valid = _mosaic_from_cfa_masked(cfa, pattern)
    green = _green_interpolation(mosaic, mask, pattern, sigma)
    red = _residual_interpolation(green, mosaic[..., 0], mask[..., 0])
    blue = _residual_interpolation(green, mosaic[..., 2], mask[..., 2])
    return np.dstack([red, green, blue])


def _igri1(
    mpfa: np.ndarray,
    mask_0: np.ndarray,
    mask_45: np.ndarray,
    mask_90: np.ndarray,
    mask_135: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width, channels = mpfa.shape
    dem_0 = np.zeros((height, width, channels), dtype=np.float64)
    dem_45 = np.zeros_like(dem_0)
    dem_90 = np.zeros_like(dem_0)
    dem_135 = np.zeros_like(dem_0)

    fn = np.array([[1 / 8, 1 / 4, 1 / 8], [1 / 8, 1 / 4, 1 / 8], [0, 0, 0]], dtype=np.float64)
    fs = np.array([[0, 0, 0], [1 / 8, 1 / 4, 1 / 8], [1 / 8, 1 / 4, 1 / 8]], dtype=np.float64)
    fw = fn.T
    fe = fs.T

    hn = np.array([[-0.5, 1.0, -0.5], [0.5, -1.0, 0.5], [0, 0, 0]], dtype=np.float64)
    hs = np.array([[0, 0, 0], [0.5, -1.0, 0.5], [-0.5, 1.0, -0.5]], dtype=np.float64)
    hw = hn.T
    he = hs.T

    mn = np.array(
        [
            [1 / 15, 1 / 15, 1 / 15, 1 / 15, 1 / 15],
            [1 / 15, 1 / 15, 1 / 15, 1 / 15, 1 / 15],
            [1 / 15, 1 / 15, 1 / 15, 1 / 15, 1 / 15],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )
    ms = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1 / 15, 1 / 15, 1 / 15, 1 / 15, 1 / 15],
            [1 / 15, 1 / 15, 1 / 15, 1 / 15, 1 / 15],
            [1 / 15, 1 / 15, 1 / 15, 1 / 15, 1 / 15],
        ],
        dtype=np.float64,
    )
    mw = mn.T
    me = ms.T

    for channel in range(channels):
        plane = mpfa[..., channel]
        xn = imfilter(plane, fn)
        xe = imfilter(plane, fe)
        xw = imfilter(plane, fw)
        xs = imfilter(plane, fs)

        inorth = np.abs(imfilter(plane, hn))
        ieast = np.abs(imfilter(plane, he))
        iwest = np.abs(imfilter(plane, hw))
        isouth = np.abs(imfilter(plane, hs))

        wn = imfilter(inorth, mn)
        we = imfilter(ieast, me)
        ww = imfilter(iwest, mw)
        ws = imfilter(isouth, ms)

        weights = _normalized_inverse_weights((wn, we, ww, ws))
        guide = weights[0] * xn + weights[1] * xe + weights[2] * xw + weights[3] * xs

        dem_90[..., channel] = _residual_interpolation(guide, mask_90 * plane, mask_90)
        dem_45[..., channel] = _residual_interpolation(guide, mask_45 * plane, mask_45)
        dem_135[..., channel] = _residual_interpolation(guide, mask_135 * plane, mask_135)
        dem_0[..., channel] = _residual_interpolation(guide, mask_0 * plane, mask_0)

    return dem_0, dem_45, dem_90, dem_135


def _igri2(
    mpfa: np.ndarray,
    mask_0: np.ndarray,
    mask_45: np.ndarray,
    mask_90: np.ndarray,
    mask_135: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width, channels = mpfa.shape
    dem_0 = np.zeros((height, width, channels), dtype=np.float64)
    dem_45 = np.zeros_like(dem_0)
    dem_90 = np.zeros_like(dem_0)
    dem_135 = np.zeros_like(dem_0)

    even_shape, odd_shape, even, odd, even_r, even_c, odd_r, odd_c = diagonal_maps(height, width)
    pattern = "rggb"
    pattern2 = "grbg"

    for channel in range(channels):
        plane = mpfa[..., channel]
        input_90_0 = np.zeros(even_shape, dtype=np.float64)
        input_135_45 = np.zeros(odd_shape, dtype=np.float64)
        valid_90_0 = np.zeros(even_shape, dtype=bool)
        valid_135_45 = np.zeros(odd_shape, dtype=bool)
        input_90_0[even_r, even_c] = plane[even]
        input_135_45[odd_r, odd_c] = plane[odd]
        valid_90_0[even_r, even_c] = True
        valid_135_45[odd_r, odd_c] = True

        id_90 = _residual_interpolation2(input_90_0, pattern, DEFAULT_SIGMA, valid_90_0)
        id_0 = _residual_interpolation2(input_90_0, pattern2, DEFAULT_SIGMA, valid_90_0)
        id_135 = _residual_interpolation2(input_135_45, pattern, DEFAULT_SIGMA, valid_135_45)
        id_45 = _residual_interpolation2(input_135_45, pattern2, DEFAULT_SIGMA, valid_135_45)

        i_90 = np.zeros((height, width), dtype=np.float64)
        i_0 = np.zeros_like(i_90)
        i_135 = np.zeros_like(i_90)
        i_45 = np.zeros_like(i_90)
        i_90[even] = id_90[even_r, even_c]
        i_0[even] = id_0[even_r, even_c]
        i_135[odd] = id_135[odd_r, odd_c]
        i_45[odd] = id_45[odd_r, odd_c]

        s0_90_0 = (i_90 + i_0) / 2.0
        s0_135_45 = (i_135 + i_45) / 2.0
        input2_90 = i_90 + s0_135_45
        input2_0 = i_0 + s0_135_45
        input2_135 = i_135 + s0_90_0
        input2_45 = i_45 + s0_90_0

        i_90_2 = _residual_interpolation2(input2_90, pattern2, DEFAULT_SIGMA)
        i_0_2 = _residual_interpolation2(input2_0, pattern2, DEFAULT_SIGMA)
        i_135_2 = _residual_interpolation2(input2_135, pattern, DEFAULT_SIGMA)
        i_45_2 = _residual_interpolation2(input2_45, pattern, DEFAULT_SIGMA)

        guide = (i_90_2 + i_0_2 + i_135_2 + i_45_2) / 4.0
        dem_90[..., channel] = _residual_interpolation(guide, mask_90 * plane, mask_90)
        dem_45[..., channel] = _residual_interpolation(guide, mask_45 * plane, mask_45)
        dem_135[..., channel] = _residual_interpolation(guide, mask_135 * plane, mask_135)
        dem_0[..., channel] = _residual_interpolation(guide, mask_0 * plane, mask_0)

    return dem_0, dem_45, dem_90, dem_135


def _color_polar_mosaic_to_bgr(
    raw_float: np.ndarray,
    bayer_pattern: str,
    sigma: float = DEFAULT_SIGMA,
) -> np.ndarray:
    height, width = raw_float.shape
    img_mpfa_bgr = np.zeros((height, width, 3), dtype=np.float64)
    phases = (
        (slice(0, None, 2), slice(0, None, 2)),
        (slice(0, None, 2), slice(1, None, 2)),
        (slice(1, None, 2), slice(0, None, 2)),
        (slice(1, None, 2), slice(1, None, 2)),
    )

    for row_slice, col_slice in phases:
        cfa = raw_float[row_slice, col_slice]
        rgb = _demosaic_bayer_cfa(cfa, bayer_pattern, sigma)
        img_mpfa_bgr[row_slice, col_slice, :] = rgb[..., ::-1]

    return img_mpfa_bgr
