from typing import Tuple, Union, Optional
import numpy as np
import cv2
import matplotlib


def applyColorMap(x: np.ndarray, colormap: Union[str, np.ndarray], vmin: float = 0.0, vmax: float = 255.0) -> np.ndarray:
    """Apply a matplotlib colormap on a given array

    Parameters
    ----------
    x : np.ndarray
        Input array
    colormap : Union[str, np.ndarray]
        Colormap name to apply, see alos matplotlib official page https://matplotlib.org/stable/gallery/color/colormap_reference.html
    vmin : float, optional
        The minimum value to normalize, by default 0.0
    vmax : float, optional
        The maximum value to normalize, by default 255.0

    Returns
    -------
    x_color : np.ndarray
        Colormapped source array. The last channel is applied color. dtype is `np.uint8`

    Examples
    --------
    Colormap from matplotlib

    >>> x = 2 * np.random.rand(256, 256) - 1  # [-1.0, 1.0]
    >>> x.shape
    (256, 256)
    >>> x_colored = applyColorMap(x, "RdBu", vmin=-1.0, vmax=1.0)
    >>> x_colored.shape
    (256, 256, 3)
    >>> x_colored.dtype
    np.uint8

    Colormap from user defined array

    >>> custom_colormap = np.zeros((256, 3), dtype=np.uint8)
    >>> custom_colormap[:128] = np.linspace(1, 0, 128)[..., None] * np.array([0, 0, 255])
    >>> custom_colormap[128:] = np.linspace(0, 1, 128)[..., None] * np.array([0, 255, 0])
    >>> x_colored = applyColorMap(x, custom_colormap, vmin=-1.0, vmax=1.0)
    """
    # Normalize the input array
    x_normalized = np.clip((x - vmin) / (vmax - vmin), 0.0, 1.0)  # [0.0, 1.0]
    x_normalized_u8 = (255 * x_normalized).astype(np.uint8)  # [0, 255]

    # Get colormap
    if isinstance(colormap, (str, matplotlib.colors.Colormap)):
        # from matplotlib
        cmap = matplotlib.cm.get_cmap(colormap, 256)
        lut = cmap(range(256))  # [0.0, 1.0], (256, 4), np.float64, RGBA
        lut = lut[:, :3]  # [0.0, 1.0], (256, 3), np.float64, RGB
        lut = lut[:, ::-1]  # [0.0, 1.0], (256, 3), np.float64, BGR
        lut_u8 = np.clip(255 * lut, 0, 255).astype(np.uint8)  # [0, 255], (256, 3), np.uint8, BGR
    elif isinstance(colormap, np.ndarray) and colormap.shape == (256, 3) and colormap.dtype == np.uint8:
        # from user defined array
        lut_u8 = colormap
    else:
        raise TypeError(f"'colormap' must be 'str' or 'np.ndarray ((256, 3), np.uint8)'.")

    x_colored = lut_u8[x_normalized_u8]  # [0, 255], BGR

    return x_colored


def applyColorToAoLP(aolp: np.ndarray, saturation: Union[float, np.ndarray] = 1.0, value: Union[float, np.ndarray] = 1.0) -> np.ndarray:
    """Apply colormap to AoLP. The colormap is based on HSV.

    Parameters
    ----------
    AoLP : np.ndarray
        AoLP, its shape is (height, width). The range is from 0.0 to pi
    saturation : Union[float, np.ndarray], optional
        Saturation value(s), by default 1.0
    value : Union[float, np.ndarray], optional
        Value value(s), by default 1.0

    Returns
    -------
    aolp_colored : np.ndarray
        An applied colormap to AoLP, its shape is (height, width, 3) and dtype is `np.uint8`
    """
    ones = np.ones_like(aolp)

    hue = (np.mod(aolp, np.pi) / np.pi * 179).astype(np.uint8)  # [0, pi] to [0, 179]
    saturation = np.clip(ones * saturation * 255, 0, 255).astype(np.uint8)
    value = np.clip(ones * value * 255, 0, 255).astype(np.uint8)

    hsv = cv2.merge([hue, saturation, value])
    aolp_colored = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return aolp_colored

