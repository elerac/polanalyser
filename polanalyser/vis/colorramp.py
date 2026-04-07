import numpy as np
import numpy.typing as npt


def linear_srgb_to_oklab(rgb):
    rgb = np.asarray(rgb)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

    l_ = np.cbrt(l)
    m_ = np.cbrt(m)
    s_ = np.cbrt(s)

    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return np.stack([L, a, b], axis=-1)


def oklab_to_linear_srgb(lab):
    lab = np.asarray(lab)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b

    l = l_**3
    m = m_**3
    s = s_**3

    r = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    return np.stack([r, g, b], axis=-1)


def srgb_to_linear_srgb(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)


def linear_srgb_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    a = 0.055
    return np.where(x <= 0.0031308, 12.92 * x, (1 + a) * (x ** (1 / 2.4)) - a)


def eval_colorramp(t: npt.ArrayLike, colors: npt.ArrayLike, space: str = "srgb") -> np.ndarray:
    """Evaluate a color ramp defined by control colors.

    This function is analogous to "gradient"/"color ramp" widgets in common
    graphics tools: it takes a set of control (anchor) colors and returns the
    interpolated RGB color(s) at position(s) ``t`` in [0, 1].

    Parameters
    ----------
    t : array_like
        Ramp position(s) in [0, 1]. Can be a scalar or an array of any shape.
    colors : array_like, (n_colors, 3)
        Control colors in RGB order. Each channel must be in [0, 1].
        At least two control colors are required.
    space : {"srgb", "oklab"}, optional
        Color space used for interpolation.
        - "srgb": Interpolate directly in sRGB (simple but not perceptually uniform).
        - "oklab": Convert control colors to Oklab, interpolate there, then convert back to sRGB (more perceptually uniform).

    Returns
    -------
    np.ndarray
        Interpolated colors in RGB order in [0, 1], with shape ``(..., 3)`` matching ``t``'s shape.
    """
    cols = np.asarray(colors)
    if cols.ndim != 2 or cols.shape[1] != 3:
        raise ValueError("colors must have shape (n_colors, 3)")
    n = cols.shape[0]
    if n < 2:
        raise ValueError("Need at least two colors")

    space = space.lower()

    # Convert control points (once per call) into interpolation space
    if space == "srgb":

        def pre_process(x: np.ndarray) -> np.ndarray:
            return x

        def post_process(x: np.ndarray) -> np.ndarray:
            return np.clip(x, 0.0, 1.0)

    elif space == "oklab":

        def pre_process(x: np.ndarray) -> np.ndarray:
            return np.asarray([linear_srgb_to_oklab(srgb_to_linear_srgb(c)) for c in x])

        def post_process(x: np.ndarray) -> np.ndarray:
            lin = oklab_to_linear_srgb(x)
            lin = np.clip(lin, 0.0, 1.0)
            return np.clip(linear_srgb_to_srgb(lin), 0.0, 1.0)

    else:
        raise ValueError(f"Unknown method={space!r}. Use 'srgb' or 'oklab'.")

    # Pre-process control colors
    ctrl = pre_process(cols)

    # Segment selection: i in [0, n-2], with t==1 mapped to last segment
    scaled = np.clip(t, 0.0, 1.0) * (n - 1)
    i = np.minimum(scaled.astype(np.int32), n - 2)

    # Local interpolation parameter u in [0, 1]
    u = scaled - i
    c0 = ctrl[i]  # shape (..., 3)
    c1 = ctrl[i + 1]  # shape (..., 3)
    interp = c0 + (c1 - c0) * u[..., None]

    return post_process(interp)
